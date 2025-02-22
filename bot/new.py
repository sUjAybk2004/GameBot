import random
import json
import pickle
import numpy as np
import tensorflow as tf
import nltk
import nlpaug.augmenter.word as naw
nltk.download('punkt')
nltk.download('wordnet')
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from nltk.stem import WordNetLemmatizer
import sys

print(sys.executable)

# Load intents
lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())

words = []
classes = []
documents = []
ignoreLetters = ['?', '!', '.', ',']

# Preprocess intents
for intent in intents['intents']:
    for pattern in intent['patterns']:
        wordList = nltk.word_tokenize(pattern)
        words.extend(wordList)
        documents.append((wordList, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize and sort words
words = [lemmatizer.lemmatize(word) for word in words if word not in ignoreLetters]
words = sorted(set(words))
classes = sorted(set(classes))

# Save words and classes
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Augment raw text data
synonym_aug = naw.SynonymAug(aug_src='wordnet')  # Use WordNet for synonym replacement

def augment_data(texts, labels, num_augments=1):
    augmented_texts = []
    augmented_labels = []

    for text, label in zip(texts, labels):
        augmented_texts.append(text)  # Keep the original text
        augmented_labels.append(label)  # Keep the original label

        # Generate augmented texts
        for _ in range(num_augments):
            augmented_text = synonym_aug.augment(text)
            augmented_texts.append(augmented_text)
            augmented_labels.append(label)  # Same label for augmented text

    return augmented_texts, augmented_labels

# Extract raw text patterns and labels
raw_texts = [pattern for intent in intents['intents'] for pattern in intent['patterns']]
raw_labels = [intent['tag'] for intent in intents['intents'] for _ in intent['patterns']]

# Apply augmentation
augmented_texts, augmented_labels = augment_data(raw_texts, raw_labels, num_augments=2)

# Rebuild documents with augmented data
augmented_documents = []
for text, label in zip(augmented_texts, augmented_labels):
    if isinstance(text, list):  # Ensure text is a string
        text = " ".join(text)
    wordList = nltk.word_tokenize(text)  # Tokenize the text
    augmented_documents.append((wordList, label))

# Rebuild training data
training = []
outputEmpty = [0] * len(classes)

for document in augmented_documents:
    bag = []
    wordPatterns = document[0]
    wordPatterns = [lemmatizer.lemmatize(word.lower()) for word in wordPatterns]
    for word in words:
        bag.append(1) if word in wordPatterns else bag.append(0)
    outputRow = list(outputEmpty)
    outputRow[classes.index(document[1])] = 1
    training.append(bag + outputRow)

random.shuffle(training)
training = np.array(training)

trainX = training[:, :len(words)]
trainY = training[:, len(words):]

# Define the model
model = Sequential()

# Input Layer
model.add(Dense(128, input_shape=(len(trainX[0]),), activation='relu', kernel_regularizer=l2(0.001)))
model.add(BatchNormalization())
model.add(Dropout(0.3))

# Hidden Layer 1
model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.001)))
model.add(BatchNormalization())
model.add(Dropout(0.3))

# Output Layer
model.add(Dense(len(trainY[0]), activation='softmax'))

# Compile the model
adam = Adam(learning_rate=0.001)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

# Early Stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
hist = model.fit(
    trainX, trainY, 
    epochs=100, batch_size=16, verbose=1, 
    validation_split=0.2, 
    callbacks=[early_stopping]
)

# Save the model
model.save('botv5.keras')
print('Executed')