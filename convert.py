import pandas as pd
import gensim.downloader as api
import numpy as np

# Load pre-trained Word2Vec model
model = api.load('word2vec-google-news-300')

def get_email_embedding(email):
    if (not isinstance(email, str)):
        email = ""
    tokens = email.split()
    word_vectors = [model[token] for token in tokens if token in model]
    
    if not word_vectors:
        return np.zeros(model.vector_size)
    
    return np.mean(word_vectors, axis=0)

if __name__=="__main__":
    email_dataset = "enron_spam_data_preprocessed.csv"
    df = pd.read_csv(email_dataset)

    X = np.array(df['Message'].apply(get_email_embedding).tolist())
    y = df["Spam/Ham"].values

    np.save("numerical_features.npy", X)
    model.save("word2vec_model.model")
