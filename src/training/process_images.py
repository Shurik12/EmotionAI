from typing import List
from tqdm import tqdm
import cv2
import os
import numpy as np
from facenet_pytorch import MTCNN

mtcnn = MTCNN(keep_all = False, post_process = False, min_face_size = 40, device = "cpu")

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
	input = 'images/'
	output = 'processed_images/'
	files = os.listdir(input)
	for i in tqdm(range(len(files))):
		file = files[i]
		image = cv2.imread(os.path.join(input, file))
		image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		face = recognize_faces(image_rgb)[0]
		cv2.imwrite(os.path.join(output, file) , cv2.cvtColor(face, cv2.COLOR_RGB2BGR))

if __name__ == '__main__':
	main()