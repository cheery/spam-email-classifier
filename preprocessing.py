import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
import string

# Download the NLTK stopwords
nltk.download('stopwords')
nltk.download('punkt')

# Open up the email dataset
email_dataset = "enron_spam_data.csv"
df = pd.read_csv(email_dataset)

def preprocess_text(text):
    # Convert text to lowercase
    if not isinstance(text, str):
        text = ""

    text = text.lower()
    
    # Remove punctuations
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Stemming
    stemmer = SnowballStemmer('english')
    tokens = [stemmer.stem(token) for token in tokens]
    
    return ' '.join(tokens)

if __name__=="__main__":
    # Apply the preprocessing function to each email in the dataframe
    df['Message'] = df['Message'].apply(preprocess_text)

    with open("enron_spam_data_preprocessed.csv", "w") as fd:
        fd.write(df.to_csv(index_label = "Message ID"))
