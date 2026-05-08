# Sign Language Detector AI

A real-time, end-to-end Machine Learning pipeline that detects hand gestures via webcam and translates them into both text and spoken audio using a modern web interface.

## Project Architecture

1. **Data Collection (`1_collect_data.py`)**: Uses MediaPipe Tasks API to extract hand landmarks and normalizes them for translation-invariance. Captures and saves custom datasets locally to CSV.
2. **Model Training (`2_train_model.py`)**: Trains a Scikit-Learn Random Forest classifier on the normalized hand landmarks to achieve high accuracy gesture recognition.
3. **Web Application (`app.py`)**: A full-stack Flask web application that streams the live webcam feed, performs real-time gesture inference, applies temporal smoothing for stability, and serves a modern Glassmorphism UI.
4. **Browser TTS**: Uses the native Web Speech API to provide seamless text-to-speech output.

## Setup Instructions

1. Install Python 3.9+
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the data collection script to record your own signs (press keys 0-4 to record 150+ frames per sign):
   ```bash
   python 1_collect_data.py
   ```
4. Train the model:
   ```bash
   python 2_train_model.py
   ```
5. Start the web application:
   ```bash
   python app.py
   ```
6. Open your web browser and navigate to `http://localhost:5000`.

## Target Gestures (Default Configuration)
- Hello
- Yes
- No
- Thank You
- Help
