"""
user_history.py — Persistent per-user review history manager.

Tracks how many reviews each user_id has submitted and how many
were flagged as fake. If a user has >50% fake reviews, they are
flagged as a malicious user.
"""

import json
import os
from datetime import date

HISTORY_FILE = os.path.join(os.path.dirname(__file__), "user_history.json")
IP_TRACKER_FILE = os.path.join(os.path.dirname(__file__), "ip_tracker.json")


def _load_json(filepath):
    if not os.path.exists(filepath):
        return {}
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}

def _save_json(data, filepath):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _load_history():
    return _load_json(HISTORY_FILE)

def _save_history(history):
    _save_json(history, HISTORY_FILE)


def get_user_history(user_id):
    """
    Get the review history for a specific user_id.

    Returns:
        dict with keys: reviews_submitted, fake_count, last_seen
        or None if user not found.
    """
    if not user_id or str(user_id).strip() == "":
        return None
    history = _load_history()
    return history.get(str(user_id).strip(), None)


def update_user_history(user_id, is_fake):
    """
    Update the review history for a user after a prediction.

    Args:
        user_id: string user identifier
        is_fake: bool — whether the review was classified as fake/manipulative
    """
    if not user_id or str(user_id).strip() == "":
        return
    
    user_id = str(user_id).strip()
    history = _load_history()

    if user_id not in history:
        history[user_id] = {
            "reviews_submitted": 0,
            "fake_count": 0,
            "last_seen": None
        }

    history[user_id]["reviews_submitted"] += 1
    if is_fake:
        history[user_id]["fake_count"] += 1
    history[user_id]["last_seen"] = str(date.today())

    _save_history(history)

def verify_ip_integrity(ip, user_id):
    """
    Tracks if the same IP is being used by multiple distinct user IDs.
    Returns: bool (True if suspected manipulation, False if safe)
    """
    if not ip or not user_id:
        return False
        
    ip = str(ip).strip()
    user_id = str(user_id).strip()
    
    ips = _load_json(IP_TRACKER_FILE)
    
    if ip not in ips:
        ips[ip] = []
        
    if user_id not in ips[ip]:
        ips[ip].append(user_id)
        _save_json(ips, IP_TRACKER_FILE)
        
    # If this single IP has been used by more than 1 distinct user ID, flag it
    return len(ips[ip]) > 1


def is_malicious_user(user_id):
    """
    Check if a user is flagged as malicious.
    A user is malicious if they have submitted at least 2 reviews
    and more than 50% were flagged as fake.

    Returns:
        tuple: (is_malicious: bool, history: dict or None)
    """
    hist = get_user_history(user_id)
    if hist is None:
        return False, None
    if hist["reviews_submitted"] < 2:
        return False, hist
    ratio = hist["fake_count"] / hist["reviews_submitted"]
    return ratio > 0.5, hist
