import whatsinmypics
from whatsinmypics import app, model
from flask import render_template, request, make_response 
import json
import os
import numpy as np
import base64

@app.route("/")
@app.route("/index.html")
@app.route("/index")
def front_page():
    return render_template('index.html', title="WhatsInMy.pics")

@app.route('/classify', methods=['POST'])
def classify():
    image = request.files['file']
    if image and valid_filename(image.filename):
        try:
            return make_response(json.dumps(model.classify_image(image)))
        except IOError:
            return "Error: invalid file"
    else:
        return "Error"

@app.route('/random', methods=['GET'])
def random_example():
    return make_response(json.dumps(model.random_image()))

@app.route('/search')
def search():
    tags = request.values.getlist('tags[]')
    classification_vector = request.values.get('classification_vector')
    classification = np.frombuffer(base64.b64decode(classification_vector), np.float32)
    return make_response(json.dumps(model.predict_images(tags, classification)))

@app.route("/slides")
def slides():
    return render_template("slides.html", title="WhatsInMy.pics")

def valid_filename(filename):
    valid_ext = [".jpg",".png",".gif",".jpeg"]
    return os.path.splitext(filename)[-1] in valid_ext
