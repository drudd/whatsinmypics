#!/usr/bin/evn python

from datetime import datetime
import sqlalchemy as sa
import itertools
import numpy as np
import pandas as pd
import pickle
from collections import defaultdict

from gensim import corpora, models 

from clean import *
from test_model import *

sample_size = 100000
timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

# load tag data
engine = sa.create_engine("mysql+pymysql://drudd:@localhost/yfcc")
labels = pd.read_sql("select * from placesCNN_labels", engine, index_col="label_index")
photos = pd.read_sql("""select yfcc.photo_id, tags,""" + ",".join("Label{}".format(i) for i in range(len(labels))) + " " +
                   """from yfcc
                      inner join placesCNN on yfcc.photo_id = placesCNN.photo_id order by rand() limit {}""".format(sample_size),
                   engine, index_col="photo_id")
photos.columns = ["tags"]+["class:"+l for l in labels['label'].tolist()]
photos['tags'] = photos['tags'].str.replace("+", " ")
photos['tags'] = photos['tags'].str.split(",")

# write out sample of photo_ids selected
pd.DataFrame(photos.index).to_csv("output/sample_{}-{}.csv".format(sample_size, timestamp))

# clean tags (independent of frequency threshold)
clean_user_tags(photos)

for frequency_threshold in [0.01, 0.0075, 0.005]:
    print "frequency_threshold:", frequency_threshold
    user_tags = filter_user_tags(photos, frequency_threshold)
    print "num_user_tags", len(user_tags)

    for classification_threshold in np.linspace(0.1, 0.6, 5):
        print "classification_threshold", classification_threshold
        class_tags = filter_class_tags(photos, classification_threshold)
        print "num_class_tags", len(class_tags)

        with open("output/tags_{}_{}_{}-{}.dat".format(frequency_threshold, classification_threshold, sample_size, timestamp), "w") as f:
            pickle.dump([user_tags,class_tags], f)

        # construct tag mapping vectors
        id2word = {i:t for i, t in enumerate(list(class_tags)+list(user_tags))}
        word2id = {t:i for i, t in id2word.items()}

        # split into train/test/validate samples after filtering
        train = photos[0:int(0.6*len(photos))]
        xval = photos[int(0.6*len(photos)):int(0.8*len(photos))]
        test = photos[int(0.8*len(photos)):]

        # find all images with both a user and classification tag
        sample = train[(train['user_tags'].apply(len) > 0) &
                       (train['class_tags'].apply(len) > 0)].copy()
        print "training photos:", len(sample)

        # construct bow representation
        sample['corpus'] = sample.apply(lambda x: [[word2id[t], 1] for t in x['user_tags'].tolist()+x['class_tags'].tolist()], axis=1)

        for passes in [10]:
            print "passes", passes

            for num_topics in [25,50,75]:
                print "num_topics", num_topics

                lda = models.LdaMulticore(sample['corpus'],
                                          id2word=id2word,
                                          num_topics=num_topics,
                                          passes=passes, workers=8)
                lda.save("output/lda_[{},{},{},{},{}]-{}".format(sample_size,
                                                                 frequency_threshold,
                                                                 classification_threshold,
                                                                 passes, num_topics,
                                                                 timestamp))

                corpus = xval[(xval['user_tags'].apply(len) > 0) &
                              (xval['class_tags'].apply(len) > 0)].apply(lambda x: [[word2id[t], 1] 
                                                                         for t in x['user_tags'].tolist()+x['class_tags'].tolist()], axis=1)

                result = test_model(lda, corpus, id2word, np.linspace(0.05, 0.95, 19))
                with open("output/model_test_[{},{},{},{},{}]-{}.dat".format(sample_size,
                                                                             frequency_threshold,
                                                                             classification_threshold,
                                                                             passes, num_topics,
                                                                             timestamp), "w") as f:
                    pickle.dump(result, f)
