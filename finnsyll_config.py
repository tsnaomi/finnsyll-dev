import os

from datetime import timedelta

local_database = 'postgresql:///finnsyll'

PERMANENT_SESSION_LIFETIME = timedelta(days=4)
SECRET_KEY = os.environ.get('SECRET_KEY') or '31415926535'
SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL', local_database)
TESTING = not bool(os.environ.get('DATABASE_URL', 0))
