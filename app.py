import pandas as pd
import pickle
import os
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Load Model
MODEL_FILE = 'model.pkl'
if os.path.exists(MODEL_FILE):
    with open(MODEL_FILE, 'rb') as f:
        model = pickle.load(f)
else:
    model = None
    print("WARNING: model.pkl not found. Please train the model first.")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded', 'prediction': 'Error', 'confidence': 0})
        
    try:
        data = request.get_json()
        landmarks_data = data.get('landmarks')
        
        if not landmarks_data or len(landmarks_data) != 63:
            return jsonify({'error': 'Invalid landmarks data', 'prediction': 'Error', 'confidence': 0})

        # Feature names must match exactly what was seen during training
        if hasattr(model, 'feature_names_in_'):
            columns = model.feature_names_in_
        else:
            columns = [f'x_{i}' for i in range(21)] + [f'y_{i}' for i in range(21)] + [f'z_{i}' for i in range(21)]
            
        df_features = pd.DataFrame([landmarks_data], columns=columns)
        
        predicted_probs = model.predict_proba(df_features)[0]
        max_prob_index = predicted_probs.argmax()
        max_prob = predicted_probs[max_prob_index]
        predicted_label = model.classes_[max_prob_index]
        
        return jsonify({
            'prediction': predicted_label,
            'confidence': float(max_prob)
        })
    except Exception as e:
        return jsonify({'error': str(e), 'prediction': 'Error', 'confidence': 0})

if __name__ == '__main__':
    # 0.0.0.0 allows access from local network devices
    app.run(host='0.0.0.0', port=5000, debug=True)
