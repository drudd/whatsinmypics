#!/usr/bin/env python

import os
import sys
from scipy.io import loadmat
import sqlalchemy as sa
import numpy as np
import pandas as pd
import caffe

# parameters of conv. neural network
MODELNAME = "placesCNN"
MODELDIR = "../models/placesCNN/"
MODEL_FILE = MODELDIR + "places205CNN_deploy_upgraded.prototxt"
PRETRAINED_FILE = MODELDIR + "places205CNN_iter_300000_upgraded.caffemodel"
MEAN = MODELDIR + "places_mean.mat"
IMG_DIM = 256
NUM_LABELS = 205

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

# initialize classifier
net = caffe.Classifier(MODEL_FILE,
                       PRETRAINED_FILE,
                       mean=mean_image,
                       channel_swap=(2,1,0),
                       raw_scale=255,
                       image_dims=(IMG_DIM, IMG_DIM))

# load sample for classification
engine = sa.create_engine("mysql+pymysql://drudd:@localhost/yfcc")

# skip images that have already been trained
trained = {int(r[0]):1 for r in engine.execute("select photo_id from placesCNN")}

count = 0
for i, (dirpath, dirnames, filenames) in enumerate(os.walk("images")):
    image_files = [filename for filename in filenames 
                   if filename.endswith(".jpg") and not int(os.path.basename(filename)[:-4]) in trained]
    if len(image_files) == 0:
        continue

    try:
        index = [int(os.path.basename(filename)[:-4]) for filename in image_files]
        images = [caffe.io.load_image(os.path.join(dirpath, filename)) for filename in image_files]

        predictions = pd.DataFrame(net.predict(images), index=index)
        predictions.index.name = "photo_id"
        top = predictions.apply(lambda r: r.argmax(), axis=1)
        predictions.columns = ["Label{}".format(i) for i in range(NUM_LABELS)]
        predictions.insert(0, 'top', top)
    
        predictions.to_sql(MODELNAME, engine, if_exists="append",
                           dtype={"top":sa.types.Integer})
        count += len(images)

        if count % 1000 == 0:
            print count
    except:
        continue
