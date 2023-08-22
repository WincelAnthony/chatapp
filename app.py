import random
import re
import json
import mysql.connector
import torch
from flask import Flask, jsonify, request
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

# Initialize Flask app
app = Flask(__name__)

# Load the trained model and data
FILE = "data.torch"
data = torch.load(FILE)
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

# Configure MySQL database connection
db_config = {
    "host": "localhost",
    "user": "root",
    "password": "",
    "database": "intents"
}

# Define function to get response from model and database
def get_response(msg):
    # Connect to database
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor()

    # Check if message is a math expression
    if re.match(r'^\d+(\.\d+)?\s*[-+*/]\s*\d+(\.\d+)?$', msg):
        try:
            result = eval(msg)
            return str(result)
        except:
            return "I'm sorry, but I couldn't calculate that expression."

    # Tokenize message and get bag of words
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    # Get model prediction
    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    # Get model confidence
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    # Query database for responses
    query = "SELECT response FROM intents WHERE tag = %s"
    cursor.execute(query, (tag,))
    responses = cursor.fetchall()

    # Close database connection
    conn.close()

    # Return response based on model confidence
    if prob.item() > 0.75:
        return random.choice(responses)[0]
    else:
        return "I apologize for the confusion. I can assist you with a wide range of topics, including ISAT-U MC student code of discipline. Please let me know how I can help you with that specific topic."

# Define Flask route for chat requests
@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data.get('query', '')
    response = get_response(user_message)
    return jsonify({'response': response})

# Run Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)