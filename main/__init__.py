import main.routes
from flask import Flask
from main.config import configure_app

app = Flask(__name__, instance_relative_config=True)
configure_app(app)
