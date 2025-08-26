#!/usr/bin/env python3
import os
import sys
import logging

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(__file__))

from src.models.redis_manager import RedisManager
from src.models.file_processor import FileProcessor
from src.app import create_app

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
redis_manager = RedisManager()
file_processor = FileProcessor(redis_manager)

app = create_app()

# Store instances in app context
app.config['redis_manager'] = redis_manager
app.config['file_processor'] = file_processor

if __name__ == '__main__':
    logger.info("Running in development mode")
    app.run(host='0.0.0.0', port=5000, threaded=True, debug=True)
else:
    logger.info("Running in production mode")