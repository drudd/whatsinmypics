import os
from flask import Flask
from flask.ext.sqlalchemy import SQLAlchemy
from threading import Lock
import numpy as np
from scipy.io import loadmat
import caffe

app = Flask(__name__)

from taggr import views

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['SQLALCHEMY_DATABASE_URI'] = "mysql+pymysql://drudd:@localhost/flickr"
db = SQLAlchemy(app)

# initialize classifier
MODELNAME = "placesCNN"
MODELDIR = "../models/placesCNN/"
MODEL_FILE = MODELDIR + "places205CNN_deploy_upgraded.prototxt"
PRETRAINED_FILE = MODELDIR + "places205CNN_iter_300000_upgraded.caffemodel"
MEAN = MODELDIR + "/places_mean.mat"
IMG_DIM = 256

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

# load labels
result = db.engine.execute("select label_index,label from placesCNN_labels")
#labels = {r[0]:r[1] for r in result}
labels = np.array([r[1] for r in result])

# initialize classifier based on pretrained data
classifier = caffe.Classifier(MODEL_FILE,
                              PRETRAINED_FILE,
                              mean=mean_image,
                              channel_swap=(2,1,0),
                              raw_scale=255,
                              image_dims=(IMG_DIM, IMG_DIM))
classifier_lock = Lock()

def predicted_label(prediction):
#    return labels[prediction.argmax()]
    return ",".join(labels[prediction > 0.1])
