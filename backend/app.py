"""
app.py — Flask API for Fake Review Detection.

Endpoints:
    POST /predict — Accepts review data, returns prediction + confidence + reasons.
    GET  /stats   — Returns aggregated fraud analytics for the Streamlit dashboard.
    GET  /health  — Health check.
"""

import sys
import os

# Add parent directory to path so we can import backend modules
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from flask import Flask, request, jsonify
from flask_cors import CORS
from backend.predictor import predict_review
from backend.activity_log import get_all_stats

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests from Streamlit


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict if a review is Genuine, Suspicious, or Manipulative.

    Expected JSON body:
    {
        "review_text"    : "Delivery was okay, packaging normal.",
        "rating"         : 5,
        "verified"       : 1,
        "ip"             : "192.168.1.1",
        "user_id"        : "user123",
        "timestamp"      : "2026-04-18T18:30:00"      ← optional, ISO 8601
    }

    Returns JSON:
    {
        "prediction": "Genuine" | "Suspicious" | "Manipulative",
        "confidence": 0.87,
        "reasons"   : ["..."]
    }
    """
    try:
        data = request.json

        if not data or "review_text" not in data:
            return jsonify({"error": "Missing 'review_text' in request body"}), 400

        review_text     = data.get("review_text", "")
        rating          = int(data.get("rating", 3))
        verified        = int(data.get("verified", 0))
        ip              = data.get("ip", "")
        user_id         = data.get("user_id", "")
        timestamp       = data.get("timestamp") or None         # None if empty/missing

        result = predict_review(
            review_text, rating, verified, ip, user_id,
            timestamp=timestamp,
        )

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/stats', methods=['GET'])
def stats():
    """
    Return aggregated fraud analytics for the Streamlit dashboard.

    Returns JSON with:
      total, genuine, suspicious, manipulative,
      malicious_users, malicious_ips,
      top_malicious_ips, timeline, recent_entries
    """
    try:
        return jsonify(get_all_stats())
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
    print("  GET  /stats   - Fraud analytics dashboard data")
    print("  GET  /health  - Health check")
    print("=" * 60)
    app.run(host='0.0.0.0', port=5000, debug=False)
