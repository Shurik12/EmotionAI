import os
import json
import uuid
import logging
from datetime import datetime
from typing import Dict, Any, Optional

import redis
from config.settings import Config

logger = logging.getLogger(__name__)

class RedisManager:
    """Handles Redis connection and operations"""
    def __init__(self):
        self.connection = self._create_connection()
        
    def _create_connection(self) -> redis.Redis:
        """Create and test Redis connection"""
        try:
            conn = redis.Redis(
                host=Config.REDIS_HOST,
                port=Config.REDIS_PORT,
                db=Config.REDIS_DB,
                password=Config.REDIS_PASSWORD,
                decode_responses=True
            )
            conn.ping()  # Test connection
            logger.info("Successfully connected to Redis")
            return conn
        except redis.ConnectionError as e:
            logger.error(f"Failed to connect to Redis: {str(e)}")
            raise

    def set_task_status(self, task_id: str, status_data: Dict[str, Any]) -> None:
        """Update task status in Redis with expiration"""
        try:
            self.connection.setex(
                f"task:{task_id}",
                time=Config.TASK_EXPIRATION,
                value=json.dumps(status_data)
            )
        except redis.RedisError as e:
            logger.error(f"Error updating task status: {str(e)}")

    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task status from Redis"""
        try:
            status = self.connection.get(f"task:{task_id}")
            return json.loads(status) if status else None
        except (redis.RedisError, json.JSONDecodeError) as e:
            logger.error(f"Error getting task status: {str(e)}")
            return None

    def save_application(self, application_data: Dict[str, Any]) -> str:
        """Save application data to a text file"""
        application_id = str(uuid.uuid4())
        application_data.update({
            "id": application_id,
            "timestamp": datetime.now().isoformat(),
            "status": "new"
        })
        
        try:
            # Save to text file
            applications_file = os.path.join(Config.UPLOAD_FOLDER, "applications.txt")
            with open(applications_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(application_data, ensure_ascii=False) + "\n")
            
            return application_id
        except Exception as e:
            logger.error(f"Error saving application: {str(e)}")
            raise