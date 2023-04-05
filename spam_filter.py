import sys
import pickle
import numpy as np
from gensim.models import KeyedVectors
from preprocessing import preprocess_text

# Function to create the average word embeddings for the input message
def message_to_embeddings(message, word_vectors, embedding_size):
    words = message.split()
    embeddings = np.zeros(embedding_size)
    valid_words = 0

    for word in words:
        if word in word_vectors:
            embeddings += word_vectors[word]
            valid_words += 1

    if valid_words > 0:
        return embeddings / valid_words
    else:
        return embeddings

# Load the SVM model from the file
with open('svm_model.pkl', 'rb') as file:
    svm_model = pickle.load(file)

# Load the KeyedVectors object from the file
word_vectors = KeyedVectors.load("word2vec_model.model")

# Read the message from stdin
message = sys.stdin.read()

# Preprocess the message
preprocessed_message = preprocess_text(message)

# Transform the message using the KeyedVectors object
embedding_size = word_vectors.vector_size
message_features = message_to_embeddings(preprocessed_message, word_vectors, embedding_size)

# Classify the message using the SVM model
prediction = svm_model.predict([message_features])

# Output the prediction to stdout
sys.stdout.write(prediction[0])
