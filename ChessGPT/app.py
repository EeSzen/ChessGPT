from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle

# Load vectorizer and model
with open("tfidf.pkl", "rb") as f:
    tfidf = pickle.load(f)

with open("svc_model.pkl", "rb") as f:
    model = pickle.load(f)

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    moves = data.get("moves", "")

    moves_transformed = tfidf.transform([moves])
    probs = model.predict_proba(moves_transformed)[0]
    predicted_class = model.predict(moves_transformed)[0]

    predicted_class_python = int(predicted_class)
    # probs_python = {
    #     '0': float(probs[0]),
    #     '1': float(probs[1])
    #     '2': float(probs[2])
    # }

    return jsonify({
        'predicted winner': predicted_class_python,
        'probabilities': {
            '0 = Black': float(probs[0] * 100),
            '1 = Draw': float(probs[1] * 100),
            '2 = White': float(probs[2] * 100),
        }
    })

if __name__ == "__main__":
    app.run(debug=True,port=5001)
