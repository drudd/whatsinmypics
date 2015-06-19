import whatsinmypics
from whatsinmypics import app
from flask import render_template, request, make_response 
import json
import os

@app.route("/")
@app.route("/index.html")
def front_page():
    return render_template('index.html', title="WhatsInMy.Pics")

@app.route('/classify', methods=['POST'])
def classify():
    image = request.files['file']
    if image and valid_filename(image.filename):
        try:
            return make_response(json.dumps(whatsinmypics.classify_image(image)))
        except IOError:
            return "Error: invalid file"
    else:
        return "Error"

def valid_filename(filename):
    valid_ext = [".jpg",".png",".gif",".jpeg"]
    return os.path.splitext(filename)[-1] in valid_ext

@app.route('/random', methods=['GET'])
def random_example():
    return make_response(json.dumps(whatsinmypics.random_image()))
