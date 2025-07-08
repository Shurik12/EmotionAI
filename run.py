import os
import numpy as np
import cv2
from typing import List
import torch
import json  # Added for JSON support

from facenet_pytorch import MTCNN
from emotiefflib.facial_analysis import EmotiEffLibRecognizer, get_model_list

# Definitions
EMOTION_CATEGORIES = ["Злость", "Отвращение", "Страх", "Счастье", "Нейтральное", "Грусть", "Удивление"]

def detect_faces(frame: np.ndarray) -> List[np.ndarray]:
    """Detect faces in a frame using MTCNN"""
    mtcnn = MTCNN(keep_all=False, post_process=False, min_face_size=40, device="cpu")
    bounding_boxes, probs = mtcnn.detect(frame, landmarks=False)
    if probs[0] is None:
        return []
    return bounding_boxes[probs > 0.9]

def recognize_faces(frame: np.ndarray) -> List[np.ndarray]:
    """Extract facial regions from frame"""
    bounding_boxes = detect_faces(frame)
    facial_images = []
    for bbox in bounding_boxes:
        box = bbox.astype(int)
        x1, y1, x2, y2 = box[0:4]
        facial_images.append(frame[y1:y2, x1:x2, :])
    return facial_images

def predictImages():
    directory = "input_files/test/image/"
    output_directory = "output_results/test/"
    os.makedirs(output_directory, exist_ok=True)
    
    files = os.listdir(directory)
    all_results = {}  # Dictionary to store all results
    
    for file in files:
        file_path = os.path.join(directory, file)
        print("Processing file:", file_path)
        image = cv2.imread(file_path)
        if image is None:
            print(f"Warning: Could not read image {file_path}, skipping...")
            continue

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        facial_images = recognize_faces(image_rgb)
        
        if not facial_images:
            print(f"Warning: No faces detected in {file_path}, skipping...")
            continue
            
        model_name = get_model_list()[4]  # Using the 5th model from the list
        device = "cuda" if torch.cuda.is_available() else "cpu"
        fer = EmotiEffLibRecognizer(engine="onnx", model_name=model_name, device=device)
        
        emotions = []
        scores_list = []
        
        for face_img in facial_images:
            emotion, scores = fer.predict_emotions(face_img, logits=False)
            emotions.append(emotion[0])
            scores_list.append(scores[0])
        
        # Get the first face's results (assuming one face per image)
        main_emotion_idx = np.argmax(scores_list[0])
        
        result = {
            "image_file": file,
            "main_prediction": {
                "index": int(main_emotion_idx),
                "label": EMOTION_CATEGORIES[main_emotion_idx],
                "probability": f"{100*scores_list[0][main_emotion_idx]:.0f}%"
            },
            "additional_probs": {
                category: f"{100*score:.0f}%"
                for category, score in zip(EMOTION_CATEGORIES, scores_list[0])
            },
            "faces_detected": len(facial_images)
        }
        
        # Save results for this image
        all_results[file] = result
        
        # Save individual JSON file for this image
        output_filename = os.path.splitext(file)[0] + ".json"
        output_path = os.path.join(output_directory, output_filename)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=4)
        print(f"Results saved to {output_path}")
    
    # Save combined results for all images
    combined_output_path = os.path.join(output_directory, "all_results.json")
    with open(combined_output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=4)
    print(f"All results combined and saved to {combined_output_path}")
    
    return all_results

def split_video_to_images() -> None:
    directory = "input_files/test/video/"
    output_directory = "input_files/test/image/"
    os.makedirs(output_directory, exist_ok=True)
    
    files = os.listdir(directory)
    for file in files:
        file_path = os.path.join(directory, file)
        print("Processing file:", file_path)
        cap = cv2.VideoCapture(file_path)    
        if not cap.isOpened():
            raise ValueError("Could not open video")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0
        print (f"Processing video: {file}, Frames: {total_frames}, FPS: {fps}, Duration: {duration:.2f}s")
    
        frame_count = 0
        processed_count = 0
        frame_interval = max(1, total_frames // 5) if total_frames > 5 else 1
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % frame_interval == 0 or frame_count == total_frames - 1:
                try:
                    frame_filename = f"frame_{processed_count}_{file.split('.')[0]}.jpg"
                    frame_path = os.path.join(output_directory, frame_filename)
                    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    cv2.imwrite(frame_path, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
                    processed_count += 1
                    if processed_count >= 5:
                        break
                except Exception as e:
                    print(f"Error processing frame {frame_count}: {str(e)}")
                    continue        
            frame_count += 1
        cap.release()

def main() -> int:
    predictImages()
    # split_video_to_images()
    return 0

if __name__ == "__main__":
    main()