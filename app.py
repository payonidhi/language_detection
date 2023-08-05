from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load the trained model and CountVectorizer
model = joblib.load('language_detection_model.joblib')
cv = joblib.load('vectorizer.joblib')
le = joblib.load('label_encoder.joblib')

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Predict route
@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    x = cv.transform([text]).toarray()
    lang = model.predict(x)
    lang = le.inverse_transform(lang)[0]  # Get the first element of the array
    prediction = "The language is in " + lang
    return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
