import whatsinmypics
from whatsinmypics import app, db, image_url, image_filename

import os
import base64
from threading import Lock
import numpy as np
from scipy.io import loadmat
import caffe
from gensim import models, similarities
from nltk.stem import WordNetLemmatizer
from PIL import Image
import skimage
from collections import defaultdict

# load mean image
ext = os.path.splitext(app.config['CNN_MEAN_FILE'])[1]
if ext == ".mat":
    mean_image = loadmat(app.config['CNN_MEAN_FILE'])['image_mean'].mean(0).mean(0)
elif ext == ".npy":
    mean_image = np.load(app.config['CNN_MEAN_FILE']).mean(1).mean(1)
elif ext == ".binaryproto":
    blob = caffe.proto.caffe_pb2.BlobProto()
    data = open(app.config['CNN_MEAN_FILE'], 'rb').read()
    blob.ParseFromString(data)
    mean_image = np.array( caffe.io.blobproto_to_array(blob) )[0].mean(1).mean(1)
else:
    print "Invalid mean image file specification"
    sys.exit(1)

# initialize classifier based on pretrained data
classifier = caffe.Classifier(app.config['CNN_MODEL_FILE'],
                              app.config['CNN_PRETRAINED_FILE'],
                              mean=mean_image,
                              channel_swap=(2,1,0),
                              raw_scale=255,
                              image_dims=(app.config['CNN_IMG_DIM'], app.config['CNN_IMG_DIM']))
classifier_lock = Lock()

# load LDA
lda = models.LdaMulticore.load(app.config['LDA_MODEL_FILE'])
word2id = {t:i for i,t in lda.id2word.items()}

lda_index = similarities.MatrixSimilarity.load(app.config['LDA_INDEX_FILE'])
lda_index.num_best = app.config['NUM_SUGGESTED_IMAGES']
lda_index_index = np.load(app.config['LDA_INDEX_INDEX_FILE'])

lemma = WordNetLemmatizer()

# LDA configuration
classification_threshold = app.config['LDA_CLASSIFICATION_THRESHOLD']
suggestion_threshold = app.config['LDA_SUGGESTION_THRESHOLD']
topN = app.config['LDA_TOPN']

def predicted_tags(classification):
    # translate classification into tag_ids and weights
    try:
        doc = [[tag_id, int(weight/classification_threshold)] 
               for tag_id, weight in enumerate(classification) if weight > classification_threshold]
  
        # add contribution from all terms in all similar LDA topics
        tag_suggestions = defaultdict(int)
        for topic, weight in lda[doc]:
            for weight, term in lda.show_topic(topic):
                if not "class:" in term:
                    tag_suggestions[term] += weight

        # turn weights into actual suggestions and take topN values as suggestion
        return [tag for tag in sorted(tag_suggestions, key=tag_suggestions.get, reverse=True)
                if tag_suggestions[tag] > suggestion_threshold][:topN] 
    except IndexError:
        return []

def predict_images(tags, classification):
    # convert document into bow
    user_tags = [[word2id[tag], 1] for tag in [lemma.lemmatize(tag) for tag in tags]
                 if tag in word2id]
    class_tags = [[tag_id, int(weight/classification_threshold)]
                  for tag_id, weight in enumerate(classification)
                  if weight > classification_threshold]
    doc = user_tags + class_tags

    # identify similar documents
    photo_ids = [lda_index_index[doc_id] for doc_id, weight in lda_index[lda[doc]]]

    result = db.engine.execute("""select download_url from yfcc
                                  where photo_id in ({})""".format(",".join(str(pid) for pid in photo_ids)))
    return {"suggested_images": [row[0] for row in result]}

def classify_image(image):
    image_path = image.filename
    image_data = np.array(Image.open(image.stream))
    image_data = skimage.img_as_float(image_data).astype(np.float32)
    with classifier_lock:
        classification = classifier.predict([image_data])[0]
        return {"suggested_tags":predicted_tags(classification), 
                "classification_vector":base64.b64encode(np.ascontiguousarray(classification.astype(np.float32)).data),
                "image_url":image_path}

def random_image():
    result = db.engine.execute("""select photo_id from sample order by rand() limit 1""")
    photo_id = result.first()[0]

    class_columns = ",".join("Label{}".format(i) for i in range(205))
    result = db.engine.execute("""select yfcc.download_url, {}
                                  from placesCNN inner join yfcc on placesCNN.photo_id = yfcc.photo_id
                                  where yfcc.photo_id = {}""".format(class_columns, photo_id))

    row = result.first()
    download_url = row[0]
    classification = np.array(row[1:])

    return {"suggested_tags":predicted_tags(classification),
            "classification_vector":base64.b64encode(np.ascontiguousarray(classification.astype(np.float32)).data),
            "image_url":download_url}
