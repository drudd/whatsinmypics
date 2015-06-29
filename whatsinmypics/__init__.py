import os
from flask import Flask
from flask.ext.sqlalchemy import SQLAlchemy

app = Flask(__name__, instance_relative_config=True)

# Load the default configuration
#app.config.from_object('config.default')

# Load the configuration from the instance folder
app.config.from_pyfile('config.py')

# connect to the database
db = SQLAlchemy(app)

# cache number of photos
num_photos = int(db.engine.execute("select count(*) from sample").first()[0])

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
