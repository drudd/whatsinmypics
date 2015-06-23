import whatsinmypics
from whatsinmypics import app, db, image_url, image_filename

import os
import base64
from threading import Lock
import numpy as np
from scipy.io import loadmat
import caffe
from gensim import models
from PIL import Image
import skimage

# classifier configuration
MODELNAME = "placesCNN"
MODELDIR = "../models/placesCNN/"
MODEL_FILE = MODELDIR + "places205CNN_deploy_upgraded.prototxt"
PRETRAINED_FILE = MODELDIR + "places205CNN_iter_300000_upgraded.caffemodel"
MEAN = MODELDIR + "/places_mean.mat"
IMG_DIM = 256

# LDA configuration
LDAMODEL = "../models/lda/lda_[1300000,0.01,0.3,10,75]-20150619020128"

# load mean image
ext = os.path.splitext(MEAN)[1]
if ext == ".mat":
    mean_image = loadmat(MEAN)['image_mean'].mean(0).mean(0)
elif ext == ".npy":
    mean_image = np.load(MEAN).mean(1).mean(1)
elif ext == ".binaryproto":
    blob = caffe.proto.caffe_pb2.BlobProto()
    data = open(MEAN, 'rb' ).read()
    blob.ParseFromString(data)
    mean_image = np.array( caffe.io.blobproto_to_array(blob) )[0].mean(1).mean(1)
else:
    print "Invalid mean image file specification"
    sys.exit(1)

# initialize classifier based on pretrained data
classifier = caffe.Classifier(MODEL_FILE,
                              PRETRAINED_FILE,
                              mean=mean_image,
                              channel_swap=(2,1,0),
                              raw_scale=255,
                              image_dims=(IMG_DIM, IMG_DIM))
classifier_lock = Lock()

result = db.engine.execute("select label_index,label from placesCNN_labels")
labels = np.array([r[1].encode("ascii") for r in result])

def predicted_tags(classification):
    return list(labels[classification > 0.2])  

def predict_images(tags, classification_vector):
    classification = np.frombuffer(base64.decodestring(classification_vector), np.float32)

    # take the top tag
    top = classification.argmax()
    result = db.engine.execute("""select yfcc.photo_id from yfcc
                                  inner join placesCNN on yfcc.photo_id = placesCNN.photo_id
                                  where placesCNN.top = {} order by rand() limit 3""".format(top))
    result = db.engine.execute("""select photo_id from placesCNN where top = {} order by rand() limit 3""".format(top))
    return {"suggested_images": [whatsinmypics.image_url(row[0]) for row in result]}

def classify_image(image):
    if isinstance(image, int):
        image_path = image_url(image)
        image = image_filename(image)
    else:
        image_path = image.filename
        image = image.stream

    image_data = np.array(Image.open(image))
    image_data = skimage.img_as_float(image_data).astype(np.float32)
    with classifier_lock:
        classification = classifier.predict([image_data])[0]
        return {"suggested_tags":predicted_tags(classification), 
                "classification_vector":base64.b64encode(np.ascontiguousarray(classification).data),
                "image_url":image_path}

def random_image():
    result = db.engine.execute("""select photo_id from placesCNN
                                  where top in (121, 122, 47, 78, 92, 128, 137, 149, 156, 163, 171, 169, 201)
                                  order by rand() limit 1""")
    return classify_image(int(result.first()[0]))
