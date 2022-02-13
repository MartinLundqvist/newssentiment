# Setting up the REST API server
from flask import Flask, jsonify, request

# An NLTK based sentiment classifier. Need to also import nltk to access the data.path
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA

# Update the path
nltk.data.path = ["nltk_data"] + nltk.data.path

app = Flask(__name__)

print(__name__)


def getSentimentNLTKVader(sentences):
    sia = SIA()
    results = []

    for line in sentences:
        pol_score=sia.polarity_scores(line)
        results.append({'sentence': line, 'score': pol_score['compound']})

    return results

@app.route('/sentiment')
def sentiment():
    sentences = request.json['sentences']
    print(f'Received the following sentences: {sentences}')
    results = getSentimentNLTKVader(sentences)  
    return jsonify({"sentiment": results})


@app.route('/tests')
def test():
    return jsonify(request.json)

@app.route('/healthz')
def home():
    return jsonify({"message": "OK"})

if (__name__ == "__main__"):
    app.run(debug=True, host="0.0.0.0", port="8080")