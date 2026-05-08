import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pandas as pd
import pickle
import pyttsx3
import threading
import time
import os

MODEL_FILE = 'model.pkl'

if not os.path.exists(MODEL_FILE):
    print(f"Error: {MODEL_FILE} not found. Please train the model using 2_train_model.py first.")
    exit()

# Load the trained model
print("Loading model...")
with open(MODEL_FILE, 'rb') as f:
    model = pickle.load(f)

# Initialize MediaPipe Tasks API
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)
]

import pythoncom
import win32com.client
from collections import deque

# State for speech debounce
last_spoken_word = None
last_spoken_time = 0
DEBOUNCE_DELAY = 3.0  # seconds between speaking the same word
CONFIDENCE_THRESHOLD = 0.75

# Temporal Smoothing to prevent flickering
prediction_buffer = deque(maxlen=7)

def speak_async(text):
    """Run TTS safely in a separate thread using Windows SAPI directly."""
    def run_tts():
        try:
            # Initialize COM for the background thread
            pythoncom.CoInitialize()
            speaker = win32com.client.Dispatch("SAPI.SpVoice")
            speaker.Speak(text)
        except Exception as e:
            print(f"TTS Error: {e}")
            
    threading.Thread(target=run_tts, daemon=True).start()

# Open Webcam
cap = cv2.VideoCapture(0)

print("Starting real-time detection... Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame using the new Tasks API
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    detection_result = detector.detect(mp_image)

    if detection_result.hand_landmarks:
        for hand_landmarks in detection_result.hand_landmarks:
            h, w, c = frame.shape
            
            # Extract landmarks and draw
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
                
                px_x, px_y = int(landmark.x * w), int(landmark.y * h)
                points_px.append((px_x, px_y))
                cv2.circle(frame, (px_x, px_y), 5, (0, 0, 255), -1)
                
            for connection in HAND_CONNECTIONS:
                start_idx, end_idx = connection
                cv2.line(frame, points_px[start_idx], points_px[end_idx], (0, 255, 0), 2)
            
            landmarks_data = x_coords + y_coords + z_coords
            
            # Make prediction
            # Use the feature names the model was trained with to avoid mismatches
            if hasattr(model, 'feature_names_in_'):
                columns = model.feature_names_in_
            else:
                columns = [f'x_{i}' for i in range(21)] + [f'y_{i}' for i in range(21)] + [f'z_{i}' for i in range(21)]
                
            df_features = pd.DataFrame([landmarks_data], columns=columns)
            
            predicted_probs = model.predict_proba(df_features)[0]
            max_prob_index = predicted_probs.argmax()
            max_prob = predicted_probs[max_prob_index]
            predicted_label = model.classes_[max_prob_index]
            
            if max_prob >= CONFIDENCE_THRESHOLD:
                prediction_buffer.append(predicted_label)
                
                # Check if we have enough frames and they all agree
                if len(prediction_buffer) == prediction_buffer.maxlen and len(set(prediction_buffer)) == 1:
                    stable_label = prediction_buffer[0]
                    
                    # Display on screen
                    display_text = f"{stable_label} ({max_prob*100:.1f}%)"
                    cv2.putText(frame, display_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    
                    # Speech output with debounce
                    current_time = time.time()
                    if stable_label != last_spoken_word or (current_time - last_spoken_time) > DEBOUNCE_DELAY:
                        speak_async(stable_label)
                        last_spoken_word = stable_label
                        last_spoken_time = current_time
                else:
                    cv2.putText(frame, "Analyzing...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
            else:
                prediction_buffer.clear()
                cv2.putText(frame, "Uncertain", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    else:
        prediction_buffer.clear()

    cv2.imshow('Sign Language Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
