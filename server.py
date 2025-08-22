import os
import yaml
import json
import uuid
import logging
import threading
from datetime import datetime
from typing import List, Dict, Any, Optional

import cv2
import numpy as np
import torch
import redis
from flask import Flask, jsonify, request, send_from_directory, send_file
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.utils import secure_filename
from facenet_pytorch import MTCNN
from emotiefflib.facial_analysis import EmotiEffLibRecognizer, get_model_list

# Configure logging before other imports to ensure proper logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('emotion_detection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_config() -> Dict[str, Any]:
    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

CONFIG = load_config()

class Config:
    """Centralized configuration management"""
    UPLOAD_FOLDER = CONFIG['app']['upload_folder']
    RESULTS_FOLDER = CONFIG['app']['results_folder']
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
    FRONTEND_BUILD_PATH = os.path.join(os.path.dirname(__file__), 'frontend', 'build')
    
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


class FileProcessor:
    """Handles file processing operations"""
    def __init__(self, redis_manager: RedisManager):
        self.redis = redis_manager
        self.mtcnn = MTCNN(
            keep_all=Config.MTCNN_CONFIG['keep_all'],
            post_process=Config.MTCNN_CONFIG['post_process'],
            min_face_size=Config.MTCNN_CONFIG['min_face_size'],
            device=Config.MTCNN_CONFIG['device']
        )
        
    def allowed_file(self, filename: str) -> bool:
        """Check if file extension is allowed"""
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS

    def cleanup_file(self, filepath: str) -> None:
        """Safely remove a file if it exists"""
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
                logger.info(f"Cleaned up file: {filepath}")
        except Exception as e:
            logger.error(f"Error cleaning up file {filepath}: {str(e)}")

    def detect_faces(self, frame: np.ndarray) -> List[np.ndarray]:
        """Detect faces in a frame using MTCNN"""
        bounding_boxes, probs = self.mtcnn.detect(frame, landmarks=False)
        if probs[0] is None:
            return []
        return bounding_boxes[probs > 0.9]

    def recognize_faces(self, frame: np.ndarray) -> List[np.ndarray]:
        """Extract facial regions from frame"""
        bounding_boxes = self.detect_faces(frame)
        facial_images = []
        for bbox in bounding_boxes:
            box = bbox.astype(int)
            x1, y1, x2, y2 = box[0:4]
            if x1 < 0:
                x1 = 0
            if y1 < 0:
                y1 = 0
            facial_images.append(frame[y1:y2, x1:x2, :])
        return facial_images

    def process_image(self, image: np.ndarray) -> tuple[np.ndarray, Dict[str, Any]]:
        """Process an image to detect emotions"""
        try:
            # Convert to RGB for display
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            facial_images = self.recognize_faces(image_rgb)
            
            if not facial_images:
                raise ValueError("no_faces_detected")
                
            model_name = get_model_list()[4]  # Using the 5th model from the list
            device = "cuda" if torch.cuda.is_available() else "cpu"
            fer = EmotiEffLibRecognizer(engine="onnx", model_name=model_name, device=device)
            
            emotions = []
            scores_list = []
            
            for face_img in facial_images:
                emotion, scores = fer.predict_emotions(face_img, logits=False)
                emotions.append(emotion[0])
                scores_list.append(scores[0])
            
            # Get the first face's results
            main_emotion_idx = np.argmax(scores_list[0])
            
            result = {
                "main_prediction": {
                    "index": int(main_emotion_idx),
                    "label": Config.EMOTION_CATEGORIES[main_emotion_idx],
                    "probability": float(scores_list[0][main_emotion_idx])
                },
                "additional_probs": {
                    category: f"{score:.2f}"
                    for category, score in zip(Config.EMOTION_CATEGORIES, scores_list[0])
                }
            }
            
            return image_rgb, result
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            raise

    def process_image_file(self, task_id: str, filepath: str, filename: str) -> None:
        """Process an image file with the specified model"""
        self.redis.set_task_status(task_id, {
            "progress": 0,
            "message": "Processing image...",
            "error": None,
            "complete": False
        })
        
        try:
            image = cv2.imread(filepath)
            if image is None:
                raise ValueError("Could not read image")
            
            self.redis.set_task_status(task_id, {
                "progress": 50,
                "message": "Processing image...",
                "error": None,
                "complete": False
            })
            
            processed_image, result = self.process_image(image)
            
            # Save result
            result_filename = f"result_{filename}"
            result_path = os.path.join(Config.RESULTS_FOLDER, result_filename)
            cv2.imwrite(result_path, cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR))
            
            # Update status
            self.redis.set_task_status(task_id, {
                "complete": True,
                "type": "image",
                "image_url": f"/api/results/{result_filename}",
                "result": result,
                "progress": 100
            })
        except Exception as e:
            raise Exception(f"Image processing failed: {str(e)}")

    def process_video_file(self, task_id: str, filepath: str, filename: str) -> None:
        """Process a video file with the specified model"""
        self.redis.set_task_status(task_id, {
            "progress": 0,
            "message": "Processing video...",
            "error": None,
            "complete": False
        })
        
        try:
            cap = cv2.VideoCapture(filepath)
            if not cap.isOpened():
                raise ValueError("Could not open video")
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps if fps > 0 else 0
            
            logger.info(f"Processing video: {filename}, Frames: {total_frames}, FPS: {fps}, Duration: {duration:.2f}s")
            
            frame_count = 0
            processed_count = 0
            results = []
            frame_interval = max(1, total_frames // 5) if total_frames > 5 else 1
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                if frame_count % frame_interval == 0 or frame_count == total_frames - 1:
                    self.redis.set_task_status(task_id, {
                        "progress": int((frame_count / total_frames) * 100),
                        "message": f"Processing frame {frame_count + 1} of {total_frames}...",
                        "error": None,
                        "complete": False
                    })
                    
                    try:
                        processed_frame, result = self.process_image(frame)
                        
                        frame_filename = f"frame_{processed_count}_{filename.split('.')[0]}.jpg"
                        frame_path = os.path.join(Config.RESULTS_FOLDER, frame_filename)
                        cv2.imwrite(frame_path, cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR))
                        
                        results.append({
                            "frame": frame_count,
                            "image_url": f"/api/results/{frame_filename}",
                            "result": result
                        })
                        processed_count += 1
                        
                        if processed_count >= 5:
                            break
                    except Exception as e:
                        logger.warning(f"Error processing frame {frame_count}: {str(e)}")
                        continue
                        
                frame_count += 1
                
            cap.release()
            
            if not results:
                raise ValueError("No frames were processed successfully")
            
            self.redis.set_task_status(task_id, {
                "complete": True,
                "type": "video",
                "frames_processed": len(results),
                "results": results,
                "progress": 100
            })
        except Exception as e:
            raise Exception(f"Video processing failed: {str(e)}")

    def process_file(self, task_id: str, filepath: str, filename: str) -> None:
        """Process a file (image or video) to detect emotions"""
        try:
            initial_status = {
                "progress": 0,
                "message": "Начало обработки...",
                "error": None,
                "complete": False,
                "model": "emotieff",
                "model_name": "EmotiEffLib"
            }
            self.redis.set_task_status(task_id, initial_status)
            file_ext = filename.rsplit('.', 1)[1].lower()
            
            if file_ext in {'png', 'jpg', 'jpeg'}:
                self.process_image_file(task_id, filepath, filename)
            elif file_ext in {'mp4', 'avi', 'webm'}:
                self.process_video_file(task_id, filepath, filename)
            else:
                self.redis.set_task_status(task_id, {
                    "error": "Неподдерживаемый формат файла",
                    "complete": True
                })
        except Exception as e:
            logger.error(f"Error processing file {filename}: {str(e)}")
            self.redis.set_task_status(task_id, {
                "error": str(e),
                "complete": True
            })
        finally:
            self.cleanup_file(filepath)

# Initialize services
redis_manager = RedisManager()
file_processor = FileProcessor(redis_manager)

def create_app():
    app = Flask(__name__, static_folder=Config.FRONTEND_BUILD_PATH)
    CORS(app)
    
    # Ensure directories exist
    os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(Config.RESULTS_FOLDER, exist_ok=True)
    os.makedirs(Config.FRONTEND_BUILD_PATH, exist_ok=True)

    # API Routes
    @app.route('/api/upload', methods=['POST'])
    def upload_file():
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        if file and file_processor.allowed_file(file.filename):
            filename = file.filename
            task_id = str(uuid.uuid4())
            filepath = os.path.join(Config.UPLOAD_FOLDER, f"{task_id}_{filename}")
            file.save(filepath)
            
            # Process file in background thread
            thread = threading.Thread(
                target=file_processor.process_file,
                args=(task_id, filepath, filename)
            )
            thread.daemon = True
            thread.start()
            
            return jsonify({"task_id": task_id}), 202
        else:
            return jsonify({"error": "Invalid file type"}), 400

    @app.route('/api/progress/<task_id>')
    def get_progress(task_id: str):
        status = redis_manager.get_task_status(task_id)
        if status is None:
            return jsonify({"error": "Task not found"}), 404
        return jsonify(status)

    @app.route('/api/submit_application', methods=['POST'])
    def submit_application():
        try:
            application_data = request.get_json()
            if not application_data:
                return jsonify({"error": "No application data provided"}), 400
            
            application_id = redis_manager.save_application(application_data)
            return jsonify({"application_id": application_id}), 201
        except Exception as e:
            logger.error(f"Error submitting application: {str(e)}")
            return jsonify({"error": "Internal server error"}), 500

    @app.route('/api/results/<filename>')
    def serve_result(filename):
        """Serve processed result files"""
        return send_from_directory(Config.RESULTS_FOLDER, filename)

    @app.route('/api/health')
    def health_check():
        """Health check endpoint"""
        try:
            redis_manager.connection.ping()
            return jsonify({"status": "healthy"}), 200
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return jsonify({"status": "unhealthy", "error": str(e)}), 500

    # Serve React static files
    @app.route('/static/<path:filename>')
    def serve_static(filename):
        static_path = os.path.join(Config.FRONTEND_BUILD_PATH, 'static')
        if os.path.exists(os.path.join(static_path, filename)):
            return send_from_directory(static_path, filename)
        return jsonify({"error": "File not found"}), 404

    # Serve React build files (JS, CSS, etc.)
    @app.route('/<path:filename>')
    def serve_react_files(filename):
        # Don't interfere with API routes
        if filename.startswith('api/'):
            return jsonify({"error": "Not found"}), 404
        
        file_path = os.path.join(Config.FRONTEND_BUILD_PATH, filename)
        
        # If it's a file that exists, serve it
        if os.path.exists(file_path) and os.path.isfile(file_path):
            return send_from_directory(Config.FRONTEND_BUILD_PATH, filename)
        
        # For React Router - serve index.html for all other routes
        return send_from_directory(Config.FRONTEND_BUILD_PATH, 'index.html')

    # Root route - serve React app
    @app.route('/')
    def serve_root():
        return send_from_directory(Config.FRONTEND_BUILD_PATH, 'index.html')

    return app

app = create_app()

if __name__ == '__main__':
    logger.info("Running in development mode")
    app.run(host='0.0.0.0', port=5000, threaded=True, debug=True)
else:
    logger.info("Running in production mode")