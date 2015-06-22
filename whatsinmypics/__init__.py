import os
from flask import Flask
from flask.ext.sqlalchemy import SQLAlchemy

app = Flask(__name__)

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['SQLALCHEMY_DATABASE_URI'] = "mysql+pymysql://drudd:@localhost/yfcc"
db = SQLAlchemy(app)

def image_url(photo_id, ext="jpg"):
    prefix1 = str(photo_id)[-2:]
    prefix2 = str(photo_id)[-4:-2]
    return os.path.join("static", "images", prefix1, prefix2, str(photo_id)+"."+ext)

def image_filename(photo_id, ext="jpg"):
    prefix1 = str(photo_id)[-2:]
    prefix2 = str(photo_id)[-4:-2]
    return os.path.join("whatsinmypics", "static",
                        "images", prefix1, prefix2,
                        str(photo_id)+"."+ext)

from whatsinmypics import views
from whatsinmypics import model
