import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load the dataset
data = pd.read_csv("data/Language_Detection.csv", encoding='latin-1')

# Separating the independent and dependent features
X = data["Text"]
y = data["Language"]

# Converting categorical variables to numerical using LabelEncoder
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

# Creating a list for appending the preprocessed text
data_list = []

# Iterating through all the text
for text in X:
    # Removing the symbols and numbers
    text = re.sub(r'[!@#$(),n"%^*?:;~`0-9]', ' ', text)
    text = re.sub(r'[[]]', ' ', text)
    # Converting the text to lower case
    text = text.lower()
    # Appending to data_list
    data_list.append(text)

# Creating Tf-idf features
tfidf_vectorizer = TfidfVectorizer()
X = tfidf_vectorizer.fit_transform(data_list).toarray()

# Splitting the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model creation and training using Naive Bayes (MultinomialNB)
nb_model = MultinomialNB()
nb_model.fit(x_train, y_train)

# Model creation and training using Random Forest
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(x_train, y_train)

# Model creation and training using Neural Network
nn_model = MLPClassifier(random_state=42)
nn_model.fit(x_train, y_train)

# Evaluating models
nb_pred = nb_model.predict(x_test)
rf_pred = rf_model.predict(x_test)
nn_pred = nn_model.predict(x_test)

nb_accuracy = accuracy_score(y_test, nb_pred)
rf_accuracy = accuracy_score(y_test, rf_pred)
nn_accuracy = accuracy_score(y_test, nn_pred)

print("Naive Bayes Accuracy:", nb_accuracy)
print("Random Forest Accuracy:", rf_accuracy)
print("Neural Network Accuracy:", nn_accuracy)

# Save the best-performing model
if rf_accuracy >= nb_accuracy and rf_accuracy >= nn_accuracy:
    model = rf_model
elif nn_accuracy >= nb_accuracy and nn_accuracy >= rf_accuracy:
    model = nn_model
else:
    model = nb_model

# Save the final trained model and vectorizer
joblib.dump(model, 'language_detection_model.joblib')
joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.joblib')
joblib.dump(le, 'label_encoder.joblib')

print("Training completed and model saved.")
