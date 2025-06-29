# Sign Language Classification Backend

This Flask backend provides gesture classification for the sign language translation feature.

## Setup Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Model (Optional)
For testing purposes, you can generate a dummy model:
```bash
python train_model.py
```

This will create a `gesture_classifier.pkl` file with a basic Random Forest classifier trained on dummy data.

### 3. Run the Server
```bash
python app.py
```

The server will start on `http://localhost:5000`

## API Endpoints

### POST /predict
Classify hand landmarks to detect gestures.

**Request Body:**
```json
{
  "landmarks": [x1, y1, z1, x2, y2, z2, ...]  // 63 values (21 landmarks * 3 coordinates)
}
```

**Response:**
```json
{
  "gesture": "Hello",
  "gesture_id": 0,
  "confidence": 0.95
}
```

### GET /health
Check server status and model availability.

### GET /
Get API information and available endpoints.

## Supported Gestures

The current model supports the following gestures:
- Hello
- Thank You
- Yes
- No
- Please
- Sorry
- Good
- Bad
- Help
- Stop

## Notes

- The current implementation uses dummy data for testing
- For production use, you should train the model with real sign language data
- The model expects 21 hand landmarks with 3 coordinates each (x, y, z)
- CORS is enabled for frontend integration 