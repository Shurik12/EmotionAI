import os
import logging
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

app = create_app()

if __name__ == '__main__':
    app.run()