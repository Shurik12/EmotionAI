from typing import List
from tqdm import tqdm
import cv2
import os
import numpy as np
from facenet_pytorch import MTCNN

mtcnn = MTCNN(keep_all=False, post_process=False, min_face_size=40, device="cpu")


def detect_faces(frame: np.ndarray) -> List[np.ndarray]:
    """Detect faces in a frame using MTCNN"""
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
        if x1 < 0:
            x1 = 0
        if y1 < 0:
            y1 = 0
        facial_images.append(frame[y1:y2, x1:x2, :])
    return facial_images


def main():
    input_root = "images/"
    output_root = "processed_images/"
    
    # Emotion folder names
    emotion_folders = ["Anger", "Disgust", "Fear", "Happiness", "Neutral", "Sadness", "Surprise"]
    
    # Process each emotion folder
    for emotion_folder in emotion_folders:
        input_folder = os.path.join(input_root, emotion_folder)
        print (input_folder)
        output_folder = os.path.join(output_root, emotion_folder)
        
        # Skip if input folder doesn't exist
        if not os.path.exists(input_folder):
            print(f"Warning: Input folder '{input_folder}' does not exist. Skipping...")
            continue
            
        # Create output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)
        
        # Get all image files in the emotion folder
        files = [f for f in os.listdir(input_folder) 
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        
        print(f"Processing folder: {emotion_folder} ({len(files)} images)")
        
        # Process each image in the folder
        for file in tqdm(files, desc=f"Processing {emotion_folder}"):
            input_path = os.path.join(input_folder, file)
            output_path = os.path.join(output_folder, file)
            
            try:
                # Read and process image
                image = cv2.imread(input_path)
                if image is None:
                    # print(f"Warning: Could not read image {input_path}. Skipping...")
                    continue
                    
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                faces = recognize_faces(image_rgb)
                
                # Skip if no faces detected
                if len(faces) == 0:
                    # print(f"Warning: No faces detected in {input_path}. Skipping...")
                    continue
                
                # Save the first detected face
                face = faces[0]
                cv2.imwrite(output_path, cv2.cvtColor(face, cv2.COLOR_RGB2BGR))
                
            except Exception as e:
                print(f"Error processing {input_path}: {str(e)}")
                continue
    
    print("Processing complete!")


if __name__ == "__main__":
    main()