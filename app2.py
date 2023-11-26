from flask import Flask, render_template, flash, request, url_for, redirect, session, jsonify
import numpy as np
import re
import tensorflow as tf
from numpy import array
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.models import load_model

app = Flask(__name__)


def init():
    global model, graph
    graph = tf.compat.v1.get_default_graph()


@app.route('/predict', methods=['GET'])
def analysis():
    if request.is_json:
        text = request.json.get('text')
        print(text)
        sentiment = ''
        max_review_length = 500
        word_to_id = imdb.get_word_index()
        strip_special_chars = re.compile("[^A-Za-z0-9 ]+")
        text = text.lower().replace("<br />", " ")
        text = re.sub(strip_special_chars, "", text.lower())

        words = text.split()
        x_test = [[word_to_id[word] if (word in word_to_id and word_to_id[word] <= 20000) else 0 for word in words]]
        x_test = sequence.pad_sequences(x_test, maxlen=max_review_length)  # Should be same which you used for training data
        vector = np.array([x_test.flatten()])

        tf.compat.v1.disable_eager_execution()

        with graph.as_default():
            model = load_model('sentiment.h5')
            probability = model.predict(array([vector][0]))[0][0]
            class1 = (model.predict(array([vector][0]))[0][0] > 0.5).astype("int32")
        if class1 == 0:
            sentiment = 'Negative'
        else:
            sentiment = 'Positive'

        print(sentiment)
        print(probability)
        return jsonify({'probability': str(probability), 'class': sentiment})
    else:
        return jsonify({'error': 'Invalid JSON in request body'}), 400


# if __name__ == "__main__":
#     init()
#     app.run()
