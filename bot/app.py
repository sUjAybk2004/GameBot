from flask import Flask, request, jsonify, render_template
import random
import json
import numpy as np
import pickle
import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import load_model

# Initialize Flask app
app = Flask(__name__)

# Load the chatbot model and preprocessing files
lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model("botv5.keras")

# Preprocess user input
def clean_up_sentence(sentence):
    """Tokenize and lemmatize the input sentence."""
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

# Convert sentence to Bag-of-Words representation
def bag_of_words(sentence):
    """Convert a sentence into a Bag-of-Words (BoW) vector."""
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

# Predict the intent of the user input
def predict_class(sentence):
    """Predict the intent of the user input."""
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)  # Sort by probability
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

# Get a response based on the predicted intent
def get_response(intents_list, intents_json):
    """Retrieve a response based on the predicted intent."""
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            # Format the response with games listed within double commas
            if isinstance(i['responses'][0], str):
                response = i['responses'][0]
                if "Try" in response or "Play" in response or "Experience" in response:
                    # Extract the list of games and remove single quotes
                    games = response.split(":")[-1].strip().split(", ")
                    games = [game.strip("'") for game in games]  # Remove single quotes
                    # Format game names within double commas and make them clickable
                    formatted_games = ",," + ", ".join(games) + ",,"
                    formatted_response = f"{response.split(':')[0]}:<br>{formatted_games}"
                else:
                    formatted_response = response
            else:
                formatted_response = random.choice(i['responses'])
            break
    return formatted_response

# Serve the HTML page
@app.route('/')
def home():
    return render_template('index.html')

# Define the route for the chatbot API
@app.route('/chat', methods=['POST'])
def chat():
    """Handle user input and return the chatbot's response."""
    # Get user input from the request
    user_input = request.json.get('message')

    # Predict the intent and get a response
    ints = predict_class(user_input)
    res = get_response(ints, intents)

    # Return the response as JSON
    return jsonify({'response': res})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)