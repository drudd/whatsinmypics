#!/usr/bin/env python

import os
import pymysql
import urllib 
import time

# create nested set of directories to ensure fast access to file metadata
for i in range(100):
    try:
        os.mkdir(os.path.join("images","{:02d}".format(i)))
    except:
        pass
    for j in range(100):
        try:
            os.mkdir(os.path.join("images","{:02d}".format(i),"{:02d}".format(j)))
        except:
            pass


connection = pymysql.connect("localhost","drudd","","yfcc")

count = 0
with connection.cursor() as cursor:
    cursor.execute("""SELECT photo_id, download_url, extension
                      FROM yfcc""")

    for photo_id, url, ext in cursor.fetchall():
        # use least-significant digits from photo_id to determine 
        # subdirectory location
        prefix1 = str(photo_id)[-2:] 
        prefix2 = str(photo_id)[-4:-2]
        filename = os.path.join("images", prefix1, prefix2,
                                str(photo_id) + "." + ext)

        # skip previously downloaded photos
        if not os.path.exists(filename):
            try:
                # download _n version which is smaller (fits in 320x240)
                url = url.replace("."+ext, "_n."+ext)
                urllib.urlretrieve(url, filename)
            except:
                # slow rate on download failure
                time.sleep(60)
        
        count += 1
        if count % 100 == 0:
            print count
