#!/usr/bin/env python

import pymysql
import numpy as np
import pandas as pd
import itertools
from sqlalchemy import create_engine
from sqlalchemy.types import String

# set fixed column names for sql import
columns = [
    "photo_id",
    "user_id",
    "user_nick",
    "date_taken",
    "date_uploaded",
    "device",
    "title",
    "description",
    "tags",
    "machine_tags",
    "longitude",
    "latitude",
    "accuracy",
    "page_url",
    "download_url",
    "license",
    "license_url",
    "server",
    "farm",
    "secret",
    "secret_original",
    "extension",
    "media_type"
]

engine = create_engine("mysql+pymysql://drudd:@localhost/yfcc") 

# select tags according to basic classes, accepting synonyms
tag_map = {
    "water":["sea","ocean","lake","river"],
    "mountain":["mountain","mountains","mount","alp","peak"],
    "city":["city","skyline"],
    "field":["field","farm","plain","meadow"],
    "forest":["forest","jungle"],
    "road":["road","bridge","track"]
}
tagset = set(itertools.chain.from_iterable(tag_map.values()))

for d in range(10):
    # read from compressed csv file in chunks for memory efficiency
    yfcc = pd.read_csv("data/yfcc100m_dataset-{}.bz2".format(d), 
                       compression="bz2", sep='\t', header=None,
                       index_col=0, na_filter=False, names=columns,
                       parse_dates=[3], 
                       converters={"date_uploaded": lambda x: pd.to_datetime(int(x), unit='s')},
                       chunksize=100000)

    for i,chunk in enumerate(yfcc):
        # select photos with any tag in tagset
        mask = chunk['tags'].str.contains("|".join(tagset), regex=True)
    
        # limit to photos (media_type == 0)
        mask &= (chunk['media_type'] == 0)

        # use pandas to_sql to insert new records into pre-existing table
        chunk[mask].to_sql("yfcc", engine, if_exists="append", index=True)
