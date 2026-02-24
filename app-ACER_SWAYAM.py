import os
import tensorflow as tf
import sys
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from src.email_utils import assess_email

# ------------------------------
# Disable GPU (always use CPU)
# ------------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.get_logger().setLevel("ERROR")

print("TensorFlow devices:", tf.config.list_physical_devices())

# ------------------------------
# NLTK Setup
# ------------------------------
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

# ------------------------------
# Flask App Init
# ------------------------------
app = Flask(__name__)

base_dir = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(base_dir, "models", "saved_models", "fake_news_detector.h5")
tokenizer_path = os.path.join(base_dir, "models", "saved_models", "tokenizer.pickle")

model = None
tokenizer = None

# ------------------------------
# Load Model
# ------------------------------
try:
    model = load_model(model_path)
    print("Model loaded.")
except Exception as e:
    print("Failed to load model:", e)

# ------------------------------
# Load Tokenizer
# ------------------------------
try:
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)
    print("Tokenizer loaded.")
except Exception as e:
    print("Failed to load tokenizer:", e)


# ------------------------------
# Text Preprocessing Function
# ------------------------------
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ------------------------------
# Probability to Label Mapping
# ------------------------------
def map_prediction(prob, threshold=0.50):
    label = "Real News" if prob >= threshold else "Fake News"

    # Calculate proper confidence
    confidence = prob if label == "Real News" else (1 - prob)
    confidence = round(float(confidence) * 100, 2)

    return label, confidence


# ------------------------------
# Routes
# ------------------------------
@app.route("/")
def home():
    return render_template("index.html", prediction=None, confidence=None)


@app.route("/predict", methods=["POST"])
def predict():
    if model is None or tokenizer is None:
        return render_template("index.html", prediction="Model or tokenizer missing.", confidence=None)

    input_text = request.form["text"]
    cleaned = preprocess_text(input_text)

    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=100, padding="post")

    raw_prob = float(model.predict(padded)[0][0])

    print("\n------------------ DEBUG ------------------")
    print("Original Text:", input_text)
    print("Cleaned:", cleaned)
    print("Sequence:", seq)
    print("Padded (first 20 tokens):", padded[0][:20])
    print("Model Raw Probability:", raw_prob)
    print("-------------------------------------------\n")

    label, confidence = map_prediction(raw_prob)

    return render_template("index.html", prediction=label, confidence=confidence)


# ------------------------------
# Email UI Route
# ------------------------------
@app.route("/email_check", methods=["GET"])
def email_check_form():
    return render_template("email_check.html")


# ------------------------------
# Email API
# ------------------------------
@app.route("/api/email_check", methods=["POST"])
def email_check_api():
    data = request.json or request.form
    from_addr = data.get("from", "")
    subject = data.get("subject", "")
    body = data.get("body", "")
    result = assess_email(from_addr, subject, body)
    return jsonify(result)


# ------------------------------
# Start Server
# ------------------------------
if __name__ == "__main__":
    if model is None or tokenizer is None:
        print("Critical component missing. Server will not start.")
        sys.exit(1)

    print("Server running at http://127.0.0.1:8000")
    app.run(host="0.0.0.0", port=8000, debug=True)
