from flask import Flask, request, jsonify
from flask_cors import CORS  # Importing CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random

app = Flask(__name__)

# Enable CORS for all domains (use this for testing, for production use more specific rules)
CORS(app)

qa_pairs = {
    "Who are you?": "Hi! I'm Sanskar Nenawati, an AI/ML Engineer.",
    "Hi?": "Hi! Wassup.",
    "What do you do?": "I specialize in Machine Learning, Data Analytics, NLP, and Gen AI.",
    "What are your skills?": "I specialize in Python, HTML, CSS, SQL, MongoDB, and libraries like PyTorch, OpenCV, scikit-learn.",
    "Tell me about your projects": "Check out my projects at https://sanskarnenawati.info/#project-section.",
    "Where do you live ?":" I am from Bhilwara rajashan but currently I live in vadodara, gujrat "
    "Do You want to do freelance work ?": "Yess Always , i am available to do freelance work contact me at sanskarnenawati@yahoo.com"
    "How can I contact you?": "You can reach me at Sanskarnenawati@yahoo.com."
}

generic_responses = [
    "Sorry Tumahare PRASHANA KA Answer hamare pass Nhi Hai , Please Call Karein +91 9119315955",
    "Sorry, I don't have an answer for that. Can you ask something else?",
    "I'm still learning, please ask me something else.",
    "That's an interesting question! Let me think about it.",
    "I don't quite understand that, but I can help with my skills or projects!",
    "I'm not sure about that, but feel free to ask about my work!"
]

# Vectorize questions
questions = list(qa_pairs.keys())
responses = list(qa_pairs.values())
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(questions)

# Route to handle chatbot queries
@app.route('/chat', methods=['POST'])
def chatbot():
    user_input = request.json.get('message', '')
    if user_input.strip() == '':
        return jsonify({"response": "Please ask a question."})

    # Vectorize the user's input and calculate similarity with predefined questions
    user_vector = vectorizer.transform([user_input])
    similarities = cosine_similarity(user_vector, tfidf_matrix)
    
    # If there's a good match (high similarity), return the predefined response
    best_match_index = similarities.argmax()
    if similarities[0][best_match_index] > 0.5:  # Threshold for good match
        response = responses[best_match_index]
    else:
        # If there's no strong match, return a generic response
        response = random.choice(generic_responses)
    
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)
