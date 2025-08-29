#!/usr/bin/env python3
import os
import sys
import logging

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(__file__))

from src.app import create_app
from src.models.redis_manager import RedisManager
from src.models.file_processor import FileProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('emotion_detection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create app instance
app = create_app()

# Initialize and configure components
with app.app_context():
    redis_manager = RedisManager()
    file_processor = FileProcessor(redis_manager)
    
    # Store instances in app config
    app.config['redis_manager'] = redis_manager
    app.config['file_processor'] = file_processor
    
    logger.info("Application initialized with Redis and FileProcessor")

# Export the app for Gunicorn
application = app