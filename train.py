import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle

email_dataset = "enron_spam_data_preprocessed.csv"
df = pd.read_csv(email_dataset)

X = np.load("numerical_features.npy")
y = df["Spam/Ham"].values

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("training logistic regression")
from sklearn.linear_model import LogisticRegression

# Train a logistic regression model
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)

print("training naive bayes")
from sklearn.naive_bayes import GaussianNB

# Train a Gaussian Naive Bayes model
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

print("training support vector machines")
from sklearn.svm import SVC

# Train a support vector machine model
svm_model = SVC()
svm_model.fit(X_train, y_train)


def evaluate_model(model, X_test, y_test):
    # Predict the labels of the test set
    y_pred = model.predict(X_test)

    # Compute performance metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label='spam')
    recall = recall_score(y_test, y_pred, pos_label='spam')
    f1 = f1_score(y_test, y_pred, pos_label='spam')

    return accuracy, precision, recall, f1

# Evaluate the Logistic Regression model
lr_accuracy, lr_precision, lr_recall, lr_f1 = evaluate_model(lr_model, X_test, y_test)

# Evaluate the Naive Bayes model
nb_accuracy, nb_precision, nb_recall, nb_f1 = evaluate_model(nb_model, X_test, y_test)

# Evaluate the Support Vector Machines model
svm_accuracy, svm_precision, svm_recall, svm_f1 = evaluate_model(svm_model, X_test, y_test)

# Print the evaluation results
print("Logistic Regression:\nAccuracy: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, F1-score: {:.4f}".format(lr_accuracy, lr_precision, lr_recall, lr_f1))
print("\nNaive Bayes:\nAccuracy: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, F1-score: {:.4f}".format(nb_accuracy, nb_precision, nb_recall, nb_f1))
print("\nSupport Vector Machines:\nAccuracy: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, F1-score: {:.4f}".format(svm_accuracy, svm_precision, svm_recall, svm_f1))

# Save the Logistic Regression model to a file
with open('logistic_regression_model.pkl', 'wb') as file:
    pickle.dump(lr_model, file)

# Save the Naive Bayes model to a file
with open('naive_bayes_model.pkl', 'wb') as file:
    pickle.dump(nb_model, file)

# Save the Support Vector Machines model to a file
with open('svm_model.pkl', 'wb') as file:
    pickle.dump(svm_model, file)
