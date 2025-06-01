import nltk
from nltk.stem import WordNetLemmatizer
import random
import json
import string

# Download required NLTK data
nltk.download('punkt')
nltk.download('wordnet')

# Sample intents data
intents = {
    "greeting": {
        "patterns": ["hello", "hi", "hey", "good morning", "good evening"],
        "responses": ["Hello!", "Hi there!", "Hey! How can I help you?"]
    },
    "goodbye": {
        "patterns": ["bye", "see you later", "goodbye"],
        "responses": ["Goodbye!", "See you soon!", "Have a great day!"]
    },
    "thanks": {
        "patterns": ["thanks", "thank you", "thanks a lot"],
        "responses": ["You’re welcome!", "No problem!", "Happy to help!"]
    },
    "weather": {
        "patterns": ["what’s the weather", "weather today", "tell me the weather"],
        "responses": ["I can’t check real-time weather yet, but it's always a good day to learn AI!"]
    },
    "name": {
        "patterns": ["what is your name", "who are you", "your name?", "may I know your name"],
        "responses": ["I’m Jarvis, your AI assistant.", "You can call me Jarvis."]
    }
}

lemmatizer = WordNetLemmatizer()

def preprocess(sentence):
    # Tokenize and lemmatize
    tokens = nltk.word_tokenize(sentence.lower())
    return [lemmatizer.lemmatize(word) for word in tokens if word not in string.punctuation]

def predict_intent(user_input):
    tokens = preprocess(user_input)
    best_match = None
    max_overlap = 0

    for intent, data in intents.items():
        for pattern in data["patterns"]:
            pattern_tokens = preprocess(pattern)
            common_tokens = set(tokens) & set(pattern_tokens)
            overlap = len(common_tokens)

            if overlap > max_overlap:
                max_overlap = overlap
                best_match = intent

    # Return best match if overlap is significant
    return best_match if max_overlap > 0 else None

def get_response(intent):
    if intent is None:
        return "Sorry, I didn’t understand that. Can you please rephrase?"
    return random.choice(intents[intent]["responses"])

def chatbot():
    print("Jarvis: Hi! I am your chatbot. Type 'quit' to exit.")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == 'quit':
            print("Jarvis: Bye! Have a great day.")
            break
        intent = predict_intent(user_input)
        response = get_response(intent)
        print(f"Jarvis: {response}")

if __name__ == "__main__":
    chatbot()
