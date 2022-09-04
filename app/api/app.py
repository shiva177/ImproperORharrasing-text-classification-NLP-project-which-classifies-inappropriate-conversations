from flask import Flask, jsonify, request
from util import utilities

app = Flask(__name__)

@app.post('/predict')
def predict():
    data = request.json
    try:
        sample = data['text']
    except KeyError:
        return jsonify({'error': 'No text sent'})

    prediction = utilities.ml_predict(sample)

    try: 
        result = jsonify(prediction)
    except TypeError as e:
        result = jsonify({'error': str(e)})

    return result

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug = True)