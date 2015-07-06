from flask import Flask
from flask.ext.sqlalchemy import SQLAlchemy

app = Flask(__name__, instance_relative_config=True)

# Load the configuration from the instance folder
# separates development from production configuration
app.config.from_pyfile('config.py')

# connect to the database
db = SQLAlchemy(app)

# load app components after database connection available
from whatsinmypics import views
from whatsinmypics import model
