import whatsinmypics
from whatsinmypics import app, model
from flask import render_template, request, make_response, jsonify
import json
import os
import numpy as np
import base64


@app.route("/")
@app.route("/index.html")
@app.route("/index")
def front_page():
    """ Render front page template """
    return render_template('index.html', title="WhatsInMy.pics")


@app.route('/classify', methods=['POST'])
def classify():
    """ Accept user-provided image upload and classify """
    image = request.files['file']
    if image and valid_filename(image.filename):
        try:
            image_response = model.classify_image(image)
        except IOError:
            return json_error("Invalid image file or bad upload")

        image_response['classification_vector'] = \
            encode_vector(image_response['classification_vector'])
        return make_response(json.dumps(image_response))
    else:
        return json_error("Invalid image file")


@app.route('/random', methods=['GET'])
def random_example():
    """ Select random image from training set and classify """
    image_response = model.random_image()
    image_response['classification_vector'] = \
        encode_vector(image_response['classification_vector'])
    return make_response(json.dumps(image_response))


@app.route('/search')
def search():
    """ Search for relevant images based on tags and classifications """
    tags = request.values.getlist('tags[]')
    classification_vector = \
        decode_vector(request.values.get('classification_vector'))
    image_response = model.predict_images(tags, classification_vector)
    return make_response(json.dumps(image_response))


@app.route("/slides")
def slides():
    """ Render slides page """
    return render_template("slides.html", title="WhatsInMy.pics")


def valid_filename(filename):
    """ Return if a file is valid based on extension """
    valid_ext = [".jpg", ".png", ".gif", ".jpeg"]
    return os.path.splitext(filename)[-1] in valid_ext


def encode_vector(vector):
    """ Convert numpy array into base64 for json embedding """
    return base64.b64encode(np.ascontiguousarray(vector.astype(np.float32)))


def decode_vector(vector):
    """ Decode base64 encoded string into numpy float32 array """
    return np.frombuffer(base64.b64decode(vector), dtype=np.float32)


def json_error(message):
    response = jsonify(message=message)
    response.status_code = 500
    return response
