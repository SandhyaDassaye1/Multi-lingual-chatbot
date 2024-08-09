from sentence_transformers import SentenceTransformer
import numpy as np
import sys
import json

# Creating predefined questions and answers, dialogues
predefined_questions = [
    "Can I check my flight information?",
    "May I rebook my flight?",
    "How do I contact support?"
]
predefined_answers = [
    "Yes, you can check your flight information on your profile!",
    "Yes, you can rebook your flight with the rebook feature on your profile!",
    "You can ask me any support questions!"
]

# Text to numbers embedding model
embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

# Classification algorithm
def predict(input_question) -> int:
    def cosine_similarity(x, y):
        dot_product = np.dot(x, y)
        norm_x = np.linalg.norm(x)
        norm_y = np.linalg.norm(y)
        return dot_product / (norm_x * norm_y)

    similarities = [
        cosine_similarity(
            embedding_model.encode([input_question])[0],
            embedding_model.encode([q])[0]
        ) for q in predefined_questions
    ]
    best_index = np.argmax(similarities)
    return predefined_answers[best_index], similarities

# Read input question from command line arguments
input_question = sys.argv[1]
reply, similarities = predict(input_question)

# Output the reply and similarities as JSON
output = {
    "reply": reply,
    "similarities": {q: s for q, s in zip(predefined_questions, similarities)}
}
print(json.dumps(output))