import os

from datetime import timedelta

local_database = 'postgresql:///finnsyll'

PERMANENT_SESSION_LIFETIME = timedelta(days=4)
SECRET_KEY = os.environ.get('SECRET_KEY') or '31415926535'
SQLALCHEMY_DATABASE_URI = os.environ['DATABASE_URL'] or local_database
TESTING = bool(os.environ['DATABASE_URL'])
