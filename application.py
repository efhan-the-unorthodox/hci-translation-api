# application.py
from main import app     # or wherever your FastAPI() instance lives
application = app       # Gunicorn will look for "application"