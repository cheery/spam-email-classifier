# Spam Email Classifier (Study Project)

This is a study project to build a binary classification model to distinguish
spam emails from non-spam (ham) emails. The project uses Python, pandas,
scikit-learn, and gensim for data processing, feature extraction, and modeling.

The dataset used in this project is the Enron Spam dataset, available at
[https://github.com/MWiechmann/enron_spam_data](https://github.com/MWiechmann/enron_spam_data).
Download the archived `enron_spam_data.csv` file from the repository and place it in the
project directory before running the scripts.

**Note**: This project is for educational purposes and not meant for actual
use, as the `spam_filter.py` script is slow and there are existing spam filters
available. The project was developed with the assistance of GPT.

## Overview

The project consists of four main Python scripts:

1. `preprocessing.py`: Preprocesses the email dataset by converting text to lowercase, removing punctuation and stopwords, and applying stemming.
2. `convert.py`: Converts the preprocessed text into numerical features using word embeddings (Word2Vec).
3. `train.py`: Trains and evaluates supervised learning algorithms (Logistic Regression, Naive Bayes, and Support Vector Machines) on the dataset.
4. `spam_filter.py`: A script that takes an email message as input and outputs whether it's spam or ham using the trained SVM model.

## Usage

1. Run `preprocessing.py` to preprocess the email dataset:

    ```
    python preprocessing.py
    ```

2. Run `convert.py` to convert the preprocessed text into numerical features:

    ```
    python convert.py
    ```

3. Run `train.py` to train and evaluate the supervised learning algorithms:

    ```
    python train.py
    ```

4. Run `spam_filter.py` to classify an email message as spam or ham:

    ```
    echo "Your email message here" | python spam_filter.py
    ```

## Dependencies

- Python 3.7 or later
- pandas
- scikit-learn
- gensim
- nltk
