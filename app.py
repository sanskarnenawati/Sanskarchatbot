from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random

app = Flask(__name__)

# Your information paragraph
paragraph = """
My name is Sanskar Nenawati, and I am an AI/ML Engineer with expertise in machine learning, 
natural language processing (NLP), and data analytics. I have worked with Python, SQL, MongoDB, 
and libraries like PyTorch, OpenCV, and scikit-learn.I am from Bhilwara rajasthan , but currently I Live in Vadodara,Gujrat. I have also developed chatbots, and I enjoy solving 
complex data problems. You can find my projects on my portfolio website. To contact me, you can email 
me at sanskarnenawati@yahoo.com and +91 9119315955.
"""

# Split paragraph into sentences for processing
paragraph_sentences = paragraph.split(". ")

# Vectorize the paragraph sentences
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(paragraph_sentences)

# List of generic responses for fallback
generic_responses = [
    "So Sorry I Can't Answer so You get opportunity to talk Here is Mobile Number : +91 9119315955"
    "Sorry, I don't have an answer for that. Can you ask something else?",
    "I'm still learning, please ask me something else.",
    "That's an interesting question! Let me think about it.",
    "I don't quite understand that, but I can help with my skills or projects!",
    "I'm not sure about that, but feel free to ask about my work!"
]

@app.route('/chat', methods=['POST'])
def chatbot():
    user_input = request.json.get('message', '')
    if user_input.strip() == '':
        return jsonify({"response": "Please ask a question."})

    # Vectorize the user's input
    user_vector = vectorizer.transform([user_input])

    # Calculate similarity between the user's input and the paragraph sentences
    similarities = cosine_similarity(user_vector, tfidf_matrix)

    # Get the index of the most similar sentence
    best_match_index = similarities.argmax()

    # If the similarity is below a certain threshold, return a generic response
    if similarities[0][best_match_index] < 0.3:  # Adjust threshold as needed
        response = random.choice(generic_responses)
    else:
        response = paragraph_sentences[best_match_index]

    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)
