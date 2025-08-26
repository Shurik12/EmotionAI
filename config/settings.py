import os
import yaml
from typing import Dict, Any

def load_config() -> Dict[str, Any]:
    # Get the project root directory (one level up from config/)
    project_root = os.path.join(os.path.dirname(__file__), '..')
    config_path = os.path.join(project_root, 'config.yaml')
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

CONFIG = load_config()

class Config:
    """Centralized configuration management"""
    UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), '..', CONFIG['app']['upload_folder'])
    RESULTS_FOLDER = os.path.join(os.path.dirname(__file__), '..', CONFIG['app']['results_folder'])
    REDIS_HOST = os.getenv('REDIS_HOST', CONFIG['redis']['host'])
    REDIS_PORT = int(os.getenv('REDIS_PORT', CONFIG['redis']['port']))
    REDIS_DB = int(os.getenv('REDIS_DB', CONFIG['redis']['db']))
    REDIS_PASSWORD = os.getenv('REDIS_PASSWORD', CONFIG['redis']['password'])
    MAX_CONTENT_LENGTH = CONFIG['app']['max_content_length']
    ALLOWED_EXTENSIONS = set(CONFIG['app']['allowed_extensions'])
    TASK_EXPIRATION = CONFIG['app']['task_expiration']
    APPLICATION_EXPIRATION = CONFIG['app']['application_expiration']
    EMOTION_CATEGORIES = CONFIG['app']['emotion_categories']
    MTCNN_CONFIG = CONFIG['mtcnn']
    FRONTEND_BUILD_PATH = os.path.join(os.path.dirname(__file__), '..', 'frontend', 'build')