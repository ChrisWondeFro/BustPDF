from flask import Flask
from app import pdf_bp
from config import Config

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)
    app.register_blueprint(pdf_bp)

    return app
