from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Define questions and responses
qa_pairs = {
    "Who are you?": "Hi! I'm Sanskar Nenawati, an AIML Engineer.",
    "What do you do?": "I am an AIML Developer with expertise in Machine Learning, data analytics, NLP, and Gen AI.",
    "What are your skills?": "I specialize in Languages: Python, HTML, CSS, SQL, MongoDB. Tools: Amazon AWS, VS Code, Git/GitHub, PowerBI, Jupyter Notebook, PyCharm. Libraries: PyTorch, OpenCV, scikit-learn, Pandas, Numpy.",
    "Tell me about your projects": "I have worked on many projects. You can check them out on https://sanskarnenawati.info/#project-section.",
    "How can I contact you?": "You can contact me at Sanskarnenawati@yahoo.com."
}

# Prepare the vectorizer
questions = list(qa_pairs.keys())
responses = list(qa_pairs.values())
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(questions)

# Route to handle chatbot queries
@app.route('/chat', methods=['POST'])
def chatbot():
    user_input = request.json.get('message', '').lower()
    
    # Vectorize the user's input
    user_vector = vectorizer.transform([user_input])
    
    # Calculate cosine similarity
    similarities = cosine_similarity(user_vector, tfidf_matrix)
    best_match_index = similarities.argmax()
    best_match_score = similarities[0, best_match_index]

    # Define a threshold for similarity
    threshold = 0.3
    if best_match_score < threshold:
        response = "I'm sorry, I didn't understand that. Can you please rephrase?"
    else:
        response = responses[best_match_index]
    
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)
