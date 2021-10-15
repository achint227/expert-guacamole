from nltk.sentiment import SentimentIntensityAnalyzer as sia
from nltk import download
from flask import Flask, request, jsonify
from waitress import serve
import logging

app = Flask(__name__)


def sentiment_analysis(text):
    return sia().polarity_scores(text)


@app.route("/")
def my_form():
    return jsonify(dict(message="Hello"))


@app.route("/", methods=["POST"])
def index():
    request_data = request.get_json()
    text = request_data["text"]
    response = sentiment_analysis(text)
    return jsonify(response)


if __name__ == "__main__":
    download('vader_lexicon')
    # app.run(debug=True,port=8080)
    serve(app, listen="*:5001")
