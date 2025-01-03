from flask import Flask, request, jsonify
from flask_cors import CORS  # Importing CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random

app = Flask(__name__)

# Enable CORS for all domains (use this for testing, for production use more specific rules)
CORS(app)

# Combined Q&A pairs including paragraph-based ones
qa_pairs = {
    "Who are you?": "Hi! I'm Sanskar Nenawati, an AI/ML Engineer. You can think of me as a problem-solving superhero, just without the cape. 🦸‍♂️",
    "Hi?": "Hi! Wassup. Ready to dive into some cool AI talk? 👾",
    "What do you do?": "I specialize in Machine Learning, Data Analytics, NLP, and Gen AI. In other words, I make machines smarter than me... and that’s saying something! 💻",
    "What are your skills?": "I specialize in Python, HTML, CSS, SQL, MongoDB, and libraries like PyTorch, OpenCV, scikit-learn. Basically, if it's tech-related, I probably know it... or at least Google it. 😉",
    "Tell me about your projects": "Check out my projects at https://sanskarnenawati.info/#project-section. Prepare to be amazed, or at least mildly impressed. 😏",
    "Where do you live?": "I am from Bhilwara, Rajasthan, but currently I live in Vadodara, Gujarat. Yes, I’m on a ‘let’s explore India’ mission, one city at a time. 🌍",
    "Do you want to do freelance work?": "Yes, always! I am available to do freelance work. Contact me at sanskarnenawati@yahoo.com. Let’s build something awesome... or at least something functional. 😅",
    "How can I contact you?": "You can reach me at Sanskarnenawati@yahoo.com. I promise I’ll reply, unless I’m busy being a coding wizard. 🧙‍♂️",
    "What is machine learning?": "Machine Learning is like teaching a machine to recognize patterns, kinda like how you learn not to step on Legos... the hard way. 👣🧩",
    "What is NLP?": "NLP stands for Natural Language Processing, and it’s how computers understand and respond to human language. So yes, I’m teaching machines to understand *your* sarcasm! 🤖",
    "What is data analytics?": "Data Analytics is the art of interpreting numbers and patterns. It’s like being a detective, but instead of a magnifying glass, I use SQL queries. 🕵️‍♂️",
    "What tools do you use for AI/ML?": "I use Python, PyTorch, OpenCV, and many others to build AI. If Python were a superhero, it would be my sidekick! 🦸‍♂️🐍",
    "How do you build chatbots?": "I use NLP and ML techniques, sprinkle a bit of AI magic, and BOOM! You’ve got a chatbot that can talk (sometimes too much). 🤖💬",
    "What’s your favorite project?": "My favorite project? It’s like asking a parent to pick a favorite child, but I do love building AI that helps people solve real-world problems. That’s my ‘proud parent’ moment. 🏆"
}

# Generic responses for unmatched questions
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
