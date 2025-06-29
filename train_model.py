import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_dummy_data():
    """Generate dummy training data for testing purposes"""
    np.random.seed(42)
    
    # Generate 100 samples per gesture (10 gestures)
    n_samples_per_gesture = 100
    n_gestures = 10
    n_features = 63  # 21 landmarks * 3 coordinates
    
    X = []
    y = []
    
    for gesture_id in range(n_gestures):
        # Generate random landmarks for each gesture
        for _ in range(n_samples_per_gesture):
            # Create realistic hand landmark data
            landmarks = []
            
            # Base coordinates for hand landmarks
            base_x = np.random.uniform(0, 1)
            base_y = np.random.uniform(0, 1)
            base_z = np.random.uniform(0, 0.5)
            
            # Generate 21 landmarks with some gesture-specific patterns
            for i in range(21):
                # Add some gesture-specific variations
                if gesture_id == 0:  # Hello - open palm
                    x = base_x + np.random.normal(0, 0.1)
                    y = base_y + np.random.normal(0, 0.1)
                    z = base_z + np.random.normal(0, 0.05)
                elif gesture_id == 1:  # Thank You - closed fist
                    x = base_x + np.random.normal(0, 0.05)
                    y = base_y + np.random.normal(0, 0.05)
                    z = base_z + np.random.normal(0, 0.02)
                else:  # Other gestures
                    x = base_x + np.random.normal(0, 0.08)
                    y = base_y + np.random.normal(0, 0.08)
                    z = base_z + np.random.normal(0, 0.03)
                
                landmarks.extend([x, y, z])
            
            X.append(landmarks)
            y.append(int(gesture_id))
    
    return np.array(X), np.array(y)

def train_model():
    """Train a Random Forest classifier on dummy data"""
    logger.info("Generating dummy training data...")
    X, y = generate_dummy_data()
    
    logger.info(f"Generated {len(X)} samples for {len(np.unique(y))} gestures")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train the model
    logger.info("Training Random Forest classifier...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate the model
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    logger.info(f"Training accuracy: {train_score:.4f}")
    logger.info(f"Testing accuracy: {test_score:.4f}")
    
    # Save the model
    model_path = 'gesture_classifier.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    logger.info(f"Model saved to {model_path}")
    
    return model

if __name__ == "__main__":
    train_model() 