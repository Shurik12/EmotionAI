import os
from flask import Flask
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from config.settings import Config
from src.routes.api import api_bp
from src.routes.frontend import frontend_bp

def create_app():
    app = Flask(__name__, static_folder=Config.FRONTEND_BUILD_PATH)
    
    # Configure app
    app.config['MAX_CONTENT_LENGTH'] = Config.MAX_CONTENT_LENGTH
    
    # Initialize extensions
    CORS(app)
    limiter = Limiter(
        app=app,
        key_func=get_remote_address,
        default_limits=["200 per day", "50 per hour"]
    )
    
    # Ensure directories exist
    os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(Config.RESULTS_FOLDER, exist_ok=True)
    os.makedirs(Config.FRONTEND_BUILD_PATH, exist_ok=True)

    # Register blueprints
    app.register_blueprint(api_bp)
    app.register_blueprint(frontend_bp)
    
    return app