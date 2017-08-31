import os

from datetime import timedelta

PERMANENT_SESSION_LIFETIME = timedelta(hours=1)
SECRET_KEY = os.getenv('SECRET_KEY')
SQLALCHEMY_DATABASE_URI = os.getenv('DATABASE_URL')
TESTING = os.getenv('TESTING', False)
DEBUG = True
