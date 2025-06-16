import os
import torch
import threading
import cv2
import logging
from datetime import datetime
from werkzeug.utils import secure_filename
from flask import Flask, jsonify, request, render_template
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from source.face_emotion_utils.predict import predict as face_predict
import redis
import json
import uuid
import numpy as np
from typing import List

from facenet_pytorch import MTCNN
from emotiefflib.facial_analysis import EmotiEffLibRecognizer, get_model_list

# For production server
from gevent import monkey
monkey.patch_all()

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

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'static/uploads'
RESULTS_FOLDER = 'static/results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4', 'avi', 'webm'}
MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Redis configuration
REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
REDIS_DB = int(os.getenv('REDIS_DB', 0))
REDIS_PASSWORD = os.getenv('REDIS_PASSWORD', '')

# Initialize Redis connection
try:
    redis_conn = redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        db=REDIS_DB,
        password=REDIS_PASSWORD,
        decode_responses=True
    )
    redis_conn.ping()  # Test connection
    logger.info("Successfully connected to Redis")
except redis.ConnectionError as e:
    logger.error(f"Failed to connect to Redis: {str(e)}")
    raise

# Rate limiting
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

# Create folders if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def cleanup_file(filepath):
    """Safely remove a file if it exists"""
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
            logger.info(f"Cleaned up file: {filepath}")
    except Exception as e:
        logger.error(f"Error cleaning up file {filepath}: {str(e)}")

def update_task_status(task_id, status_data):
    """Update task status in Redis with expiration"""
    try:
        redis_conn.setex(
            f"task:{task_id}",
            time=3600,  # 1 hour expiration
            value=json.dumps(status_data)
        )
    except redis.RedisError as e:
        logger.error(f"Error updating task status in Redis: {str(e)}")

def get_task_status(task_id):
    """Get task status from Redis"""
    try:
        status = redis_conn.get(f"task:{task_id}")
        return json.loads(status) if status else None
    except (redis.RedisError, json.JSONDecodeError) as e:
        logger.error(f"Error getting task status from Redis: {str(e)}")
        return None

def process_image(image, model_type="default"):
    """Process an image to detect emotions using the specified model"""
    try:
        # Convert to RGB for display
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if model_type == "emotieff":
            # Process with EmotiEffLibRecognizer
            device = "cuda" if torch.cuda.is_available() else "cpu"
            facial_images = recognize_faces(image_rgb, device)
            
            if not facial_images:
                raise ValueError("No faces detected in the image")
                
            model_name = get_model_list()[4]  # Using the 5th model from the list
            fer = EmotiEffLibRecognizer(engine="onnx", model_name=model_name, device=device)
            
            emotions = []
            scores_list = []
            categories = ["Anger", "Disgust", "Fear", "Happiness", "Neutral", "Sadness", "Surprise"]
            
            for face_img in facial_images:
                emotion, scores = fer.predict_emotions(face_img, logits=False)
                emotions.append(emotion[0])
                scores_list.append(scores[0])
            
            # Get the first face's results (we'll assume one face per image for simplicity)
            main_emotion_idx = np.argmax(scores_list[0])
            main_emotion = categories[main_emotion_idx]
            
            # Prepare additional probabilities
            additional = {
                category: f"{score:.2f}"
                for category, score in zip(categories, scores_list[0])
            }
            
            result = {
                "main_prediction": {
                    "index": int(main_emotion_idx),
                    "label": main_emotion,
                    "probability": float(scores_list[0][main_emotion_idx])
                },
                "additional_probs": additional
            }
        else:
            # Default model processing (original code)
            prediction = face_predict(image_rgb)
            
            # Prepare the additional probabilities
            additional = {
                "Sad/Fear": f"{prediction[2][0]:.4f}",
                "Neutral": f"{prediction[2][1]:.4f}",
                "Happy": f"{prediction[2][2]:.4f}",
                "Angry": f"{prediction[2][3]:.4f}",
                "Surprise/Disgust": f"{prediction[2][4]:.4f}"
            }
            
            result = {
                "main_prediction": {
                    "index": int(prediction[1]),
                    "label": prediction[0],
                    "probability": float(max(prediction[2]))
                },
                "additional_probs": additional
            }
        
        return image_rgb, result
    except Exception as e:
        logger.error(f"Error processing image with model {model_type}: {str(e)}")
        raise

def process_file(task_id, filepath, filename, model_type="default"):
    """Process a file (image or video) to detect emotions"""
    try:
        initial_status = {
            "progress": 0,
            "message": "Starting processing...",
            "error": None,
            "complete": False,
            "model": model_type,
            "model_name": "EmotiEffLib" if model_type == "emotieff" else "EmotionNet"
        }
        update_task_status(task_id, initial_status)
        
        file_ext = filename.rsplit('.', 1)[1].lower()
        
        if file_ext in {'png', 'jpg', 'jpeg'}:
            process_image_file(task_id, filepath, filename, model_type)
        elif file_ext in {'mp4', 'avi', 'webm'}:
            process_video_file(task_id, filepath, filename, model_type)
        else:
            update_task_status(task_id, {
                "error": "Unsupported file type",
                "complete": True
            })
    except Exception as e:
        logger.error(f"Error processing file {filename}: {str(e)}")
        update_task_status(task_id, {
            "error": str(e),
            "complete": True
        })
    finally:
        cleanup_file(filepath)

def process_image_file(task_id, filepath, filename, model_type="default"):
    """Process an image file with the specified model"""
    update_task_status(task_id, {
        "progress": 0,
        "message": "Processing image...",
        "error": None,
        "complete": False
    })
    
    try:
        image = cv2.imread(filepath)
        if image is None:
            raise ValueError("Could not read image")
        
        # Update progress
        update_task_status(task_id, {
            "progress": 50,
            "message": "Processing image...",
            "error": None,
            "complete": False
        })
        
        # Process image
        processed_image, result = process_image(image, model_type)
        
        # Save result
        result_filename = f"result_{filename}"
        result_path = os.path.join(app.config['RESULTS_FOLDER'], result_filename)
        cv2.imwrite(result_path, cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR))
        
        # Update status
        update_task_status(task_id, {
            "complete": True,
            "type": "image",
            "image_url": f"/{result_path}",
            "result": result,
            "progress": 100
        })
    except Exception as e:
        raise Exception(f"Image processing failed: {str(e)}")

def process_video_file(task_id, filepath, filename, model_type="default"):
    """Process a video file with the specified model"""
    update_task_status(task_id, {
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
        
        # Calculate frame interval to process (aim for 5 frames total)
        frame_interval = max(1, total_frames // 5) if total_frames > 5 else 1
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process every nth frame to get a sample
            if frame_count % frame_interval == 0 or frame_count == total_frames - 1:
                status_update = {
                    "progress": int((frame_count / total_frames) * 100),
                    "message": f"Processing frame {frame_count + 1} of {total_frames}...",
                    "error": None,
                    "complete": False
                }
                update_task_status(task_id, status_update)
                
                try:
                    processed_frame, result = process_image(frame, model_type)
                    
                    # Save frame
                    frame_filename = f"frame_{processed_count}_{filename.split('.')[0]}.jpg"
                    frame_path = os.path.join(app.config['RESULTS_FOLDER'], frame_filename)
                    cv2.imwrite(frame_path, cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR))
                    
                    results.append({
                        "frame": frame_count,
                        "image_url": f"/{frame_path}",
                        "result": result
                    })
                    processed_count += 1
                    
                    # Limit to 5 frames max
                    if processed_count >= 5:
                        break
                except Exception as e:
                    logger.warning(f"Error processing frame {frame_count}: {str(e)}")
                    continue
                    
            frame_count += 1
            
        cap.release()
        
        if not results:
            raise ValueError("No frames were processed successfully")
        
        update_task_status(task_id, {
            "complete": True,
            "type": "video",
            "frames_processed": len(results),
            "results": results,
            "progress": 100
        })
    except Exception as e:
        raise Exception(f"Video processing failed: {str(e)}")

def detect_face(frame: np.ndarray):
    device = "cpu"
    mtcnn = MTCNN(keep_all=False, post_process=False, min_face_size=40, device=device)
    bounding_boxes, probs = mtcnn.detect(frame, landmarks=False)
    if probs[0] is None:
        return []
    bounding_boxes = bounding_boxes[probs > 0.9]
    return bounding_boxes

def recognize_faces(frame: np.ndarray, device: str) -> List[np.array]:
    bounding_boxes = detect_face(frame)
    facial_images = []
    for bbox in bounding_boxes:
        box = bbox.astype(int)
        x1, y1, x2, y2 = box[0:4]
        facial_images.append(frame[y1:y2, x1:x2, :])
    return facial_images

@app.route('/')
def index():
    """Render the main landing page"""
    return render_template('landing.html')

@app.route('/detector')
def detector():
    """Render the emotion detector page"""
    return render_template('detector.html')

@app.route('/upload', methods=['POST'])
@limiter.limit("10 per minute")
def upload_file():
    if 'file' not in request.files:
        logger.warning("No file provided in upload request")
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    selected_model = request.form.get('model', 'default')
    
    if file.filename == '':
        logger.warning("Empty filename in upload request")
        return jsonify({"error": "No selected file"}), 400
    
    if not file or not allowed_file(file.filename):
        logger.warning(f"Invalid file type attempted: {file.filename}")
        return jsonify({"error": "Invalid file type. Supported types: JPG, PNG, MP4, AVI, WEBM"}), 400
    
    try:
        # Create unique task ID
        task_id = str(uuid.uuid4())
        
        # Secure filename and save file
        filename = secure_filename(f"{task_id}_{file.filename}")
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        file.save(filepath)
        logger.info(f"File uploaded successfully: {filename}")

        initial_status = {
            "progress": 0,
            "message": "Starting processing...",
            "error": None,
            "complete": False,
            "model": "default",  # Always use the same model regardless of selection
            "model_name": "EmotionNet"  # Display name for the actual model being used
        }
        update_task_status(task_id, initial_status)
        
        logger.info(f"Added task to execution: {task_id}")
        
        # Start processing in background
        thread = threading.Thread(
            target=process_file,
            args=(task_id, filepath, filename, selected_model),
            daemon=True
        )
        thread.start()
        
        return jsonify({
            "task_id": task_id,
            "message": "File uploaded and processing started",
            "model": "default",  # Inform client about the actual model being used
            "model_name": "EmotionNet"
        })
    except Exception as e:
        logger.error(f"Error during file upload: {str(e)}")
        cleanup_file(filepath)
        return jsonify({"error": f"Upload failed: {str(e)}"}), 500

@app.route('/progress/<task_id>')
def get_progress(task_id):
    """Check processing progress for a task"""
    status = get_task_status(task_id)
    if status is None:
        return jsonify({"error": "Task not found"}), 404
    return jsonify(status)

@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file size limit exceeded"""
    logger.warning("File size limit exceeded in upload request")
    return jsonify({"error": "File too large. Maximum size is 16MB"}), 413

@app.errorhandler(429)
def ratelimit_handler(e):
    """Handle rate limiting"""
    logger.warning(f"Rate limit exceeded: {str(e)}")
    return jsonify({"error": "Rate limit exceeded. Please try again later."}), 429
    
if __name__ == '__main__':
    # Development configuration
    logger.info("Running in development mode")
    app.run(host='0.0.0.0', port=5000, threaded=True)
else:
    # Production configuration
    logger.info("Running in production mode")
