from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
import logging

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the trained model
model_path = 'gesture_classifier.pkl'
if os.path.exists(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    logger.info(model,"Model loaded successfully tffffffffffffff")
else:
    logger.warning("Model file not found. Using dummy classifier.")
    model = None

# Gesture mapping (you can customize this based on your training data)
GESTURE_MAPPING = {
    0: "Hello",
    1: "Thank You", 
    2: "Yes",
    3: "No",
    4: "Please",
    5: "Sorry",
    6: "Good",
    7: "Bad",
    8: "Help",
    9: "Stop"
}

@app.route('/predict', methods=['POST'])
def predict_gesture():
    try:
        data = request.get_json()
        
        if not data or 'landmarks' not in data:
            return jsonify({'error': 'No landmarks provided'}), 400
        
        landmarks = data['landmarks']

        print("landmarks",landmarks)
        
        # Validate landmarks format
        if not landmarks or len(landmarks) == 0:  # 21 points * 3 coordinates
            return jsonify({'error': 'Invalid landmarks format. Expected 63 values (21 points x 3 coordinates)'}), 400
        
        # Convert to numpy array and reshape
        landmarks_array = np.array(landmarks).reshape(1, -1)
        
        if model is None:
            # Dummy classifier for testing
            import random
            gesture_id = random.randint(0, len(GESTURE_MAPPING) - 1)
            gesture = GESTURE_MAPPING[gesture_id]
            logger.info(f"Dummy prediction: {gesture}")
        else:
            # Make prediction with actual model
            prediction = model.predict(landmarks_array)
            gesture_id = prediction[0]
            gesture = GESTURE_MAPPING.get(gesture_id, "Unknown")
            logger.info(f"Model prediction: {gesture} (ID: {gesture_id})")
        
        return jsonify({
            'gesture': gesture,
            'gesture_id': int(gesture_id),
            'confidence': 0.95  # Dummy confidence for now
        })
        
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'available_gestures': list(GESTURE_MAPPING.values())
    })

@app.route('/', methods=['GET'])
def index():
    return jsonify({
        'message': 'Sign Language Gesture Classification API',
        'endpoints': {
            'POST /predict': 'Classify hand landmarks to gesture',
            'GET /health': 'Health check and model status',
            'GET /': 'API information'
        }
    })

if __name__ == '__main__':
    logger.info("Starting Sign Language Classification API...")
    app.run(debug=True, host='0.0.0.0', port=5000) 