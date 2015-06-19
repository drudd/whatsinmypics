import os
from flask import Flask
from flask.ext.sqlalchemy import SQLAlchemy
from threading import Lock
import numpy as np
from scipy.io import loadmat
import caffe
from gensim import models
from PIL import Image
import skimage

app = Flask(__name__)

from whatsinmypics import views

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['SQLALCHEMY_DATABASE_URI'] = "mysql+pymysql://drudd:@localhost/flickr"
db = SQLAlchemy(app)

# classifier configuration
MODELNAME = "placesCNN"
MODELDIR = "../models/placesCNN/"
MODEL_FILE = MODELDIR + "places205CNN_deploy_upgraded.prototxt"
PRETRAINED_FILE = MODELDIR + "places205CNN_iter_300000_upgraded.caffemodel"
MEAN = MODELDIR + "/places_mean.mat"
IMG_DIM = 256

# LDA configuration
LDAMODEL = "../models/lda/lda_[1000000,0.01,0.1,10,50]-20150617235806"

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
labels = np.array([r[1].encode("ascii") for r in result])

# initialize classifier based on pretrained data
classifier = caffe.Classifier(MODEL_FILE,
                              PRETRAINED_FILE,
                              mean=mean_image,
                              channel_swap=(2,1,0),
                              raw_scale=255,
                              image_dims=(IMG_DIM, IMG_DIM))
classifier_lock = Lock()


# initialize LDA model
lda = models.LdaMulticore.load(LDAMODEL)
word2id = {t:i for i, t in lda.id2word.items()}

def predicted_label(prediction):
#    return labels[prediction.argmax()]
    return list(labels[prediction > 0.2])

def predicted_images(prediction):
    # take the top tag
    top = prediction.argmax()
    result = db.engine.execute("""select filename from photos
                                  inner join placesCNN on photos.photo_id = placesCNN.photo_id
                                  where placesCNN.top = {} order by rand() limit 3""".format(top))
    return [row[0] for row in result]

def classify_image(image):
    if isinstance(image, basestring):
        input_file = "/static/"+image
        image = "whatsinmypics/static/"+image
    else:
        input_file = image.filename
        image = image.stream

    image_data = np.array(Image.open(image))
    image_data = skimage.img_as_float(image_data).astype(np.float32)
    with classifier_lock:
        prediction = classifier.predict([image_data])[0]
        label = predicted_label(prediction)
        images = ["/static/"+filename for filename in predicted_images(prediction)]
        return {"suggested_tags":label, "suggested_images":images,
                "prediction_vector":prediction.tolist(), "input_file":input_file}

def random_image():
    result = db.engine.execute("""select filename from photos
                                  inner join placesCNN on photos.photo_id = placesCNN.photo_id
                                  order by rand() limit 1""")
    return classify_image(result.first()[0]) 
