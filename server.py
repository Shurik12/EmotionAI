import os
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
from flask import Flask, jsonify, request, send_from_directory
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

# Constants
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4', 'avi', 'webm'}
MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB
EMOTION_CATEGORIES = ["Злость", "Отвращение", "Страх", "Счастье", "Нейтральное", "Грусть", "Удивление"]
TASK_EXPIRATION = 3600  # 1 hour in seconds
APPLICATION_EXPIRATION = 86400 * 30  # 30 days in seconds

class Config:
    """Centralized configuration management"""
    UPLOAD_FOLDER = 'static/uploads'
    RESULTS_FOLDER = 'static/results'
    REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
    REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
    REDIS_DB = int(os.getenv('REDIS_DB', 0))
    REDIS_PASSWORD = os.getenv('REDIS_PASSWORD', 'redis_password')

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
                time=TASK_EXPIRATION,
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
        """Save application data to Redis"""
        application_id = str(uuid.uuid4())
        application_data.update({
            "id": application_id,
            "timestamp": datetime.now().isoformat(),
            "status": "new"
        })
        
        try:
            self.connection.setex(
                f"application:{application_id}",
                time=APPLICATION_EXPIRATION,
                value=json.dumps(application_data)
            )
            return application_id
        except redis.RedisError as e:
            logger.error(f"Error saving application: {str(e)}")
            raise

class FileProcessor:
    """Handles file processing operations"""
    def __init__(self, redis_manager: RedisManager):
        self.redis = redis_manager
        self.mtcnn = MTCNN(keep_all=False, post_process=False, min_face_size=40, device="cpu")
        
    def allowed_file(self, filename: str) -> bool:
        """Check if file extension is allowed"""
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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
            facial_images.append(frame[y1:y2, x1:x2, :])
        return facial_images

    def process_image(self, image: np.ndarray) -> tuple[np.ndarray, Dict[str, Any]]:
        """Process an image to detect emotions"""
        try:
            # Convert to RGB for display
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            facial_images = self.recognize_faces(image_rgb)
            
            if not facial_images:
                raise ValueError("На изображении не обнаружено лиц")
                
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
                    "label": EMOTION_CATEGORIES[main_emotion_idx],
                    "probability": float(scores_list[0][main_emotion_idx])
                },
                "additional_probs": {
                    category: f"{score:.2f}"
                    for category, score in zip(EMOTION_CATEGORIES, scores_list[0])
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
                "image_url": f"/{result_path}",
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
                            "image_url": f"/{frame_path}",
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

def create_app():
    """Application factory pattern"""
    app = Flask(__name__, static_folder='static', template_folder='templates')
    CORS(app)
    
    # Configuration
    app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH
    app.config['RATELIMIT_HEADERS_ENABLED'] = True
    
    # Create folders if they don't exist
    os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(Config.RESULTS_FOLDER, exist_ok=True)
    
    # Initialize services
    redis_manager = RedisManager()
    file_processor = FileProcessor(redis_manager)
    
    # Rate limiting
    def limiter_breach_handler(request_limit):
        """Custom handler for rate limit breaches"""
        logger.warning(f"Rate limit breached for {request_limit.key}")
        remaining = request_limit.remaining
        limit = request_limit.limit
        
        reset_time = request_limit.reset_at
        if isinstance(reset_time, int):
            reset_time = datetime.fromtimestamp(reset_time)
        
        message = (
            f"Превышен лимит запросов ({limit}). "
            f"Попробуйте снова через {remaining} секунд. "
            f"Лимит полностью восстановится в {reset_time.strftime('%H:%M:%S')}"
        )
        
        response = jsonify({
            "error": "rate_limit_exceeded",
            "message": message,
            "limit": str(limit),
            "remaining": remaining,
            "reset_at": reset_time.isoformat() if hasattr(reset_time, 'isoformat') else reset_time
        })
        response.status_code = 429
        return response

    limiter = Limiter(
        app=app,
        key_func=get_remote_address,
        default_limits=[],
        storage_uri=f"redis://:{Config.REDIS_PASSWORD}@{Config.REDIS_HOST}:{Config.REDIS_PORT}/{Config.REDIS_DB}",
        strategy="fixed-window",
        on_breach=limiter_breach_handler
    )

    @app.route('/', defaults={'path': ''})
    @app.route('/<path:path>')
    def serve(path):
        # Handle API routes first
        if path.startswith('api/') or path in ['upload', 'progress', 'submit_application', 'health']:
            # Let Flask handle these routes normally
            return
        
        # Serve static files if they exist
        if path.startswith('static/'):
            try:
                return send_from_directory(app.static_folder, path[len('static/'):])
            except:
                pass
        
        # For all other routes, serve the index.html
        try:
            return send_from_directory(app.template_folder, 'index.html')
        except:
            return "Page not found", 404

    @app.route('/api/upload', methods=['POST'])
    @limiter.limit("5 per hour")
    def upload_file():
        if 'file' not in request.files:
            logger.warning("No file provided in upload request")
            return jsonify({"error": "Файл не предоставлен"}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            logger.warning("Empty filename in upload request")
            return jsonify({"error": "Файл не выбран"}), 400
        
        if not file or not file_processor.allowed_file(file.filename):
            logger.warning(f"Invalid file type attempted: {file.filename}")
            return jsonify({"error": "Неподдерживаемый формат файла. Поддерживаются: JPG, PNG, MP4, AVI, WEBM"}), 400
        
        try:
            task_id = str(uuid.uuid4())
            filename = secure_filename(f"{task_id}_{file.filename}")
            filepath = os.path.join(Config.UPLOAD_FOLDER, filename)
            
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            file.save(filepath)
            logger.info(f"File uploaded successfully: {filename}")

            # Start processing in background
            thread = threading.Thread(
                target=file_processor.process_file,
                args=(task_id, filepath, filename),
                daemon=True
            )
            thread.start()
            
            return jsonify({
                "task_id": task_id,
                "message": "Файл загружен, начата обработка",
                "model": "emotieff",
                "model_name": "EmotiEffLib"
            })
        except Exception as e:
            logger.error(f"Error during file upload: {str(e)}")
            file_processor.cleanup_file(filepath)
            return jsonify({"error": f"Upload failed: {str(e)}"}), 500

    @app.route('/api/progress/<task_id>')
    def get_progress(task_id: str):
        """Check processing progress for a task"""
        status = redis_manager.get_task_status(task_id)
        if status is None:
            return jsonify({"error": "Task not found"}), 404
        return jsonify(status)

    @app.route('/api/submit_application', methods=['POST'])
    @limiter.limit("10 per minute")
    def submit_application():
        try:
            data = request.get_json()
            if not data or 'plan' not in data or 'email' not in data:
                return jsonify({"error": "Недостаточно данных"}), 400
            
            application_data = {
                "plan": data.get('plan'),
                "name": data.get('name', ''),
                "email": data.get('email'),
                "phone": data.get('phone', ''),
                "company": data.get('company', ''),
            }
            
            try:
                application_id = redis_manager.save_application(application_data)
                logger.info(f"New application received: {application_id}")
                return jsonify({"success": True, "application_id": application_id})
            except redis.RedisError:
                return jsonify({"error": "Ошибка при сохранении заявки"}), 500
                
        except Exception as e:
            logger.error(f"Error processing application: {str(e)}")
            return jsonify({"error": "Ошибка при обработке заявки"}), 500

    @app.route('/api/health')
    def health_check():
        try:
            redis_manager.connection.ping()
            limiter.storage.check()
            return jsonify({"status": "healthy"}), 200
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return jsonify({"status": "unhealthy", "error": str(e)}), 500

    @app.errorhandler(413)
    def request_entity_too_large(error):
        logger.warning("File size limit exceeded in upload request")
        return jsonify({"error": "Файл слишком большой. Максимальный размер 50MB"}), 413

    @app.errorhandler(429)
    def ratelimit_handler(e):
        logger.warning(f"Rate limit exceeded: {str(e)}")
        return jsonify({
            "error": "rate_limit_exceeded",
            "message": "Превышен лимит запросов. Пожалуйста, попробуйте позже."
        }), 429
    
    return app

app = create_app()

if __name__ == '__main__':
    logger.info("Running in development mode")
    app.run(host='0.0.0.0', port=5000, threaded=True)
else:
    logger.info("Running in production mode")