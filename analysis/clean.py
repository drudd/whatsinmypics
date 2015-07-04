from nltk.stem import WordNetLemmatizer
import csv
import re
import pandas as pd
import numpy as np
from collections import defaultdict

# perform tag cleaning
def clean_user_tags(photos):
    lemma = WordNetLemmatizer()
    def clean_photo_tags(taglist):
        return [lemma.lemmatize(tag).encode('ascii') for tag in taglist
                if not "%" in tag and not "." in tag and not "\u" in tag
                and len(tag) > 1 and "\\x" not in tag and not re.match("^\d+$", tag)]
    photos['clean_user_tags'] = photos['tags'].apply(clean_photo_tags)

def filter_user_tags(photos, frequency_threshold=0.05):
    # load country, city, state names and codes
    filter_tags = set()
    with open("../tables/state_table.csv","r") as f:
        reader = csv.reader(f)
        for state in reader:
            filter_tags |= set([state[1].lower(),
                                state[1].replace(" ","").lower(),
                                state[2].lower()])

    with open("../tables/countries.txt","r") as f:
        reader = csv.reader(f)
        for country in reader:
            filter_tags |= set([country[0].lower(),
                                country[1].replace(" ","").lower(),
                                country[1].lower()])
        
    with open("../tables/cities.txt", "r") as f:
        for city in f:
            city = city.rstrip("\n")
            filter_tags |= set([city.lower(),
                                city.replace(" ", "").lower()])
        
    with open("../tables/times.txt", "r") as f:        
        for time in f:
            time = time.rstrip("\n")
            filter_tags |= set(time.split(",")) # already lower
        
    with open("../tables/camera_words.txt", "r") as f:
        for word in f:
            filter_tags |= set([word.rstrip("\n").lower()])

    # count tag frequency
    tag_count = defaultdict(int)
    for taglist in photos['clean_user_tags']:
        for tag in taglist:
            tag_count[tag] += 1

    def filter_photo_tags(taglist):
        return [tag for tag in taglist if not tag in filter_tags and tag_count[tag] > frequency_threshold*len(photos)]
    photos['user_tags'] = photos['clean_user_tags'].apply(filter_photo_tags) 

    user_tags = set()
    for taglist in photos['user_tags']:
        user_tags |= set(taglist)
    return user_tags

def filter_class_tags(photos, classification_threshold=0.2):
    classify_columns = np.array([col for col in photos.columns if "class:" in col])
    def filter_photo_tags(photo):
        return [col for i, col in enumerate(classify_columns) if photo[i] > classification_threshold]
    photos['class_tags'] = photos[classify_columns].apply(filter_photo_tags, axis=1)

    class_tags = set()
    for taglist in photos['class_tags']:
        class_tags |= set(taglist)
    return class_tags 
