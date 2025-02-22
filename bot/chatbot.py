import random
import json
import numpy as np
import pickle
import nltk
import nlpaug.augmenter.word as naw
from nltk.stem import WordNetLemmatizer
from keras.models import load_model

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

model = load_model("chatbot3.keras")

def clean_up_sentence(sentence):
    """This function is used for preprocessing user input before passing it to the chatbot model. It performs tokenization and lemmatization to convert the input sentence into a cleaned list of words."""
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words (sentence):
    """This function converts a sentence into a Bag-of-Words (BoW) representation, which is a numerical feature vector used for machine learning models. It checks whether each word in the chatbot's vocabulary (words) appears in the input sentence and creates a binary vector."""
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class (sentence):
    """This function takes a user input sentence, processes it, and predicts the most relevant intent using the trained chatbot model. It returns a list of probable intents with confidence scores."""
    bow = bag_of_words (sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)      #Sorts the intents from highest to lowest probability.
    return_list = []
    for r in results:
        return_list.append({'intent': classes [r[0]], 'probability': str(r[1])})
    # classes[r[0]] maps intent index to its name.
    # Creates a list of dictionaries, each containing:
    # 'intent': The predicted intent name.
    # 'probability': The confidence score.
    return return_list

def get_response(intents_list, intents_json):
    """This function retrieves an appropriate response based on the predicted intent. It selects a random response from a predefined intent-response dataset (intents_json)."""
    tag = intents_list[0]['intent']                     #Extracts the intent name from the first (highest confidence) prediction.
    list_of_intents = intents_json['intents']
    for i in list_of_intents:                           #Loops through the list of intents to find the one that matches tag.
        if i['tag'] == tag:
            result = random.choice (i['responses'])
            break
    return result

print("GO!!")

while True:
    """This loop creates an interactive chatbot that continuously takes user input, predicts the intent, and returns a response."""
    message = input("")
    ints = predict_class (message)
    res = get_response (ints, intents)
    print (res)