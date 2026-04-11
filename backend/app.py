"""
app.py — Flask API for Fake Review Detection.

Endpoints:
    POST /predict — Accepts review data, returns prediction + confidence + reasons.
"""

import sys
import os

# Add parent directory to path so we can import backend modules
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from flask import Flask, request, jsonify
from flask_cors import CORS
from backend.predictor import predict_review

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests from Streamlit


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict if a review is Genuine, Suspicious, or Manipulative.

    Expected JSON body:
    {
        "review_text": "This product is amazing!",
        "rating": 5,
        "verified": 1,
        "ip": "192.168.1.1",
        "user_id": "user123"
    }

    Returns JSON:
    {
        "prediction": "Genuine" | "Suspicious" | "Manipulative",
        "confidence": 0.85,
        "reasons": ["Not a verified purchase", ...]
    }
    """
    try:
        data = request.json

        if not data or "review_text" not in data:
            return jsonify({"error": "Missing 'review_text' in request body"}), 400

        review_text = data.get("review_text", "")
        rating = int(data.get("rating", 3))
        verified = int(data.get("verified", 0))
        ip = data.get("ip", "")
        user_id = data.get("user_id", "")

        result = predict_review(review_text, rating, verified, ip, user_id)

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok", "message": "Fake Review Detector API is running"})


if __name__ == '__main__':
    print("=" * 60)
    print("Fake Review Detector API")
    print("=" * 60)
    print("Endpoints:")
    print("  POST /predict - Analyze a review")
    print("  GET  /health  - Health check")
    print("=" * 60)
    app.run(host='0.0.0.0', port=5000, debug=False)
