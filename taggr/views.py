import taggr
from taggr import app
import json
from flask import render_template, request, make_response 
import os
from PIL import Image
import skimage
import numpy as np

@app.route("/")
@app.route("/index.html")
def front_page():
    return render_template('index.html', title="Taggr")

@app.route('/classify', methods=['POST'])
def classify():
    image = request.files['file']
    if image and valid_filename(image.filename):
        try:
            image_data = np.array(Image.open(image.stream))
            image_data = skimage.img_as_float(image_data).astype(np.float32)
            with taggr.classifier_lock:
                prediction = taggr.classifier.predict([image_data])[0]
                label = taggr.predicted_label(prediction)
                images = ["/static/"+filename for filename in taggr.predicted_images(prediction)]
                return make_response(json.dumps({"suggested_tags":label, "suggested_images":images})) 
        except IOError:
            return "Error: invalid file"
    else:
        return "Error"

def valid_filename(filename):
    valid_ext = [".jpg",".png",".gif",".jpeg"]
    return os.path.splitext(filename)[-1] in valid_ext
