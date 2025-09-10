import os
import logging
from typing import List, Dict, Any

import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN
from emotiefflib.facial_analysis import EmotiEffLibRecognizer, get_model_list

from config.settings import Config
from src.models.redis_manager import RedisManager

logger = logging.getLogger(__name__)

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
                
            model_name = get_model_list()[3]  # Using the 5th model from the list
            device = "cuda" if torch.cuda.is_available() else "cpu"
            fer = EmotiEffLibRecognizer(engine="onnx", model_name=model_name, device=device)
            
            emotions = []
            scores_list = []
            
            for face_img in facial_images:
                emotion, scores = fer.predict_emotions(face_img, logits=False)
                print (f'valence: {scores[0][-2]}')
                print (f'arousal: {scores[0][-1]}')
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