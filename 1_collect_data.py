import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pandas as pd
import os
import csv
import numpy as np

# Use the new MediaPipe Tasks API
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

# Hand connections for drawing
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)
]

# Gestures mapping
GESTURES = {
    '0': 'Hello',
    '1': 'Yes',
    '2': 'No',
    '3': 'Thank You',
    '4': 'Help'
}

DATASET_FILE = 'dataset.csv'

# Initialize CSV file with headers if it doesn't exist
if not os.path.exists(DATASET_FILE):
    with open(DATASET_FILE, mode='w', newline='') as f:
        writer = csv.writer(f)
        # 63 features (21 landmarks * 3 coordinates (x,y,z)) + label
        headers = [f'x_{i}' for i in range(21)] + [f'y_{i}' for i in range(21)] + [f'z_{i}' for i in range(21)] + ['label']
        writer.writerow(headers)

# Open Webcam
cap = cv2.VideoCapture(0)

print("Starting data collection...")
print("Press '0'-'4' to record a sample for the corresponding gesture.")
for k, v in GESTURES.items():
    print(f"  {k}: {v}")
print("Press 'q' to quit.")

sample_counts = {label: 0 for label in GESTURES.values()}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Ignoring empty camera frame.")
        continue

    # Flip the image horizontally for a later selfie-view display, and convert the BGR image to RGB.
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame using the new Tasks API
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    detection_result = detector.detect(mp_image)

    landmarks_data = []
    
    if detection_result.hand_landmarks:
        for hand_landmarks in detection_result.hand_landmarks:
            h, w, c = frame.shape
            
            # Extract landmarks and draw manually
            x_coords = []
            y_coords = []
            z_coords = []
            
            # Normalize by making all coordinates relative to the wrist (landmark 0)
            wrist_x = hand_landmarks[0].x
            wrist_y = hand_landmarks[0].y
            wrist_z = hand_landmarks[0].z
            
            points_px = []
            for landmark in hand_landmarks:
                x_coords.append(landmark.x - wrist_x)
                y_coords.append(landmark.y - wrist_y)
                z_coords.append(landmark.z - wrist_z)
                
                # Convert to pixel coordinates for drawing
                px_x, px_y = int(landmark.x * w), int(landmark.y * h)
                points_px.append((px_x, px_y))
                cv2.circle(frame, (px_x, px_y), 5, (0, 0, 255), -1)
            
            # Draw connections
            for connection in HAND_CONNECTIONS:
                start_idx, end_idx = connection
                cv2.line(frame, points_px[start_idx], points_px[end_idx], (0, 255, 0), 2)
            
            landmarks_data = x_coords + y_coords + z_coords

    # Display instructions and sample counts
    y_pos = 30
    cv2.putText(frame, "Press 0-4 to record, Q to quit", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    y_pos += 30
    for k, v in GESTURES.items():
        count_text = f"{v} ({k}): {sample_counts[v]}"
        cv2.putText(frame, count_text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_pos += 25

    cv2.imshow('Data Collection', frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    
    char_key = chr(key) if key < 256 else ''
    if char_key in GESTURES and landmarks_data:
        label = GESTURES[char_key]
        with open(DATASET_FILE, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(landmarks_data + [label])
        sample_counts[label] += 1
        print(f"Recorded sample for {label}. Total: {sample_counts[label]}")

cap.release()
cv2.destroyAllWindows()
