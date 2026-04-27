"""
activity_log.py — Temporal fraud activity tracker.

Logs every review prediction with timestamp, user_id, ip, and prediction.
Provides burst detection and temporal fraud analysis for post-processing overrides.
"""

import json
import os
from datetime import datetime, timedelta
from collections import defaultdict

ACTIVITY_LOG_FILE = os.path.join(os.path.dirname(__file__), "activity_log.json")


# ─── Internal JSON helpers ─────────────────────────────────────────────────────

def _load_log():
    if not os.path.exists(ACTIVITY_LOG_FILE):
        return {"entries": []}
    try:
        with open(ACTIVITY_LOG_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if "entries" in data else {"entries": []}
    except (json.JSONDecodeError, IOError):
        return {"entries": []}


def _save_log(data):
    with open(ACTIVITY_LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _parse_ts(ts_str):
    """Safely parse ISO timestamp string to datetime."""
    try:
        return datetime.fromisoformat(ts_str)
    except (ValueError, TypeError):
        return None


# ─── Core Logging ─────────────────────────────────────────────────────────────

def log_activity(user_id, ip, timestamp, prediction):
    """
    Append a review prediction event to the activity log.

    Args:
        user_id  : str — user identifier (empty string if unknown)
        ip       : str — IP address
        timestamp: str — ISO 8601 format or None (auto-fills current time)
        prediction: str — Genuine / Suspicious / Manipulative
    """
    if timestamp is None:
        timestamp = datetime.now().isoformat()

    data = _load_log()
    data["entries"].append({
        "user_id": str(user_id).strip() if user_id else "",
        "ip":      str(ip).strip()      if ip      else "",
        "timestamp": timestamp,
        "prediction": prediction,
    })

    # Cap at 10,000 entries to prevent unbounded growth
    if len(data["entries"]) > 10_000:
        data["entries"] = data["entries"][-10_000:]

    _save_log(data)


# ─── Temporal Analysis Functions ───────────────────────────────────────────────

def analyze_user_burst(user_id, current_timestamp_str=None, window_minutes=5):
    """
    Count reviews submitted by `user_id` in the last `window_minutes` minutes.

    Returns:
        (count: int, is_burst: bool)
        is_burst is True if count >= 3
    """
    if not user_id:
        return 0, False

    user_id = str(user_id).strip()
    data    = _load_log()
    
    ref_time = _parse_ts(current_timestamp_str) if current_timestamp_str else None
    if not ref_time:
        ref_time = datetime.now()
        
    cutoff  = ref_time - timedelta(minutes=window_minutes)
    count   = 1 # Include current review

    for entry in data["entries"]:
        if entry.get("user_id") == user_id:
            ts = _parse_ts(entry.get("timestamp"))
            if ts and cutoff <= ts <= ref_time:
                count += 1

    return count, count >= 3


def analyze_ip_burst(ip, current_user_id=None, current_timestamp_str=None, window_minutes=60):
    """
    Count reviews and distinct users from a given IP in the last `window_minutes` minutes.

    Returns:
        (review_count: int, unique_user_count: int, is_suspicious: bool)
        is_suspicious if review_count >= 5 AND unique_user_count >= 2
    """
    if not ip:
        return 0, 0, False

    ip       = str(ip).strip()
    data     = _load_log()
    
    ref_time = _parse_ts(current_timestamp_str) if current_timestamp_str else None
    if not ref_time:
        ref_time = datetime.now()
        
    cutoff   = ref_time - timedelta(minutes=window_minutes)
    reviews  = 1 # Include current review
    users    = set()
    if current_user_id:
        users.add(str(current_user_id).strip())

    for entry in data["entries"]:
        if entry.get("ip") == ip:
            ts = _parse_ts(entry.get("timestamp"))
            if ts and cutoff <= ts <= ref_time:
                reviews += 1
                uid = entry.get("user_id", "")
                if uid:
                    users.add(uid)

    is_suspicious = reviews >= 5 and len(users) >= 2
    return reviews, len(users), is_suspicious


def detect_midnight_spam(user_id, current_timestamp_str=None, min_reviews=3, threshold_ratio=0.60):
    """
    Check if a user posts predominantly during midnight hours (00:00–04:00).

    Returns:
        (midnight_ratio: float, is_suspicious: bool)
        Only flags if the user has at least `min_reviews` total entries.
    """
    if not user_id:
        return 0.0, False

    user_id       = str(user_id).strip()
    data          = _load_log()
    total         = 1 # Include current review
    midnight_cnt  = 0
    
    ref_time = _parse_ts(current_timestamp_str) if current_timestamp_str else None
    if ref_time and 0 <= ref_time.hour < 4:
        midnight_cnt += 1

    for entry in data["entries"]:
        if entry.get("user_id") == user_id:
            ts = _parse_ts(entry.get("timestamp"))
            if ts:
                total += 1
                if 0 <= ts.hour < 4:
                    midnight_cnt += 1

    if total < min_reviews:
        return 0.0, False

    ratio = midnight_cnt / total
    return round(ratio, 3), ratio >= threshold_ratio


# ─── Dashboard Aggregation ────────────────────────────────────────────────────

def get_all_stats():
    """
    Compute aggregated fraud intelligence for the analytics dashboard.

    Merges two data sources:
      1. dataset_stats.json  — pre-computed baseline from the full 90k dataset
                               (run backend/precompute_dataset_stats.py once to create it)
      2. activity_log.json   — real-time entries added with every new prediction

    Returns a dict with:
      total, genuine, suspicious, manipulative,
      malicious_users, malicious_ips,
      top_malicious_ips: [[ip, fake_count], ...],
      recent_entries: [{...}, ...] (last 20, newest first — real-time only)
    """
    # ── Load dataset baseline ─────────────────────────────────────────────
    dataset_stats_file = os.path.join(os.path.dirname(__file__), "dataset_stats.json")
    ds = {"total": 0, "genuine": 0, "suspicious": 0, "manipulative": 0, "top_malicious_ips": []}
    try:
        with open(dataset_stats_file, "r", encoding="utf-8") as f:
            loaded = json.load(f)
            ds.update(loaded)
    except Exception:
        pass  # dataset_stats.json not yet generated — show real-time only

    # ── Load real-time activity log ───────────────────────────────────────
    data    = _load_log()
    entries = data.get("entries", [])

    rt_genuine      = sum(1 for e in entries if e.get("prediction") == "Genuine")
    rt_suspicious   = sum(1 for e in entries if e.get("prediction") == "Suspicious")
    rt_manipulative = sum(1 for e in entries if e.get("prediction") == "Manipulative")

    # ── Merge totals ──────────────────────────────────────────────────────
    total        = ds["total"]        + len(entries)
    genuine      = ds["genuine"]      + rt_genuine
    suspicious   = ds["suspicious"]   + rt_suspicious
    manipulative = ds["manipulative"] + rt_manipulative

    # ── Merge malicious IP counts ─────────────────────────────────────────
    ip_fake_counts = defaultdict(int)

    # Start with dataset IPs
    for item in ds.get("top_malicious_ips", []):
        ip, cnt = item[0], item[1]
        if ip:
            ip_fake_counts[ip] += cnt

    # Layer real-time flagged IPs on top
    for e in entries:
        ip = e.get("ip", "")
        if ip and e.get("prediction") in ("Suspicious", "Manipulative"):
            ip_fake_counts[ip] += 1

    top_malicious_ips = sorted(
        ip_fake_counts.items(), key=lambda x: x[1], reverse=True
    )[:10]

    # ── Malicious users (from user_history.json) ──────────────────────────
    user_history_file    = os.path.join(os.path.dirname(__file__), "user_history.json")
    malicious_user_count = 0
    try:
        with open(user_history_file, "r", encoding="utf-8") as f:
            history = json.load(f)
        for uid, hist in history.items():
            submitted = hist.get("reviews_submitted", 0)
            if submitted >= 2 and hist.get("fake_count", 0) / submitted > 0.5:
                malicious_user_count += 1
    except Exception:
        pass

    malicious_ip_count = len([ip for ip, cnt in ip_fake_counts.items() if cnt >= 2])

    return {
        "total":             total,
        "genuine":           genuine,
        "suspicious":        suspicious,
        "manipulative":      manipulative,
        "malicious_users":   malicious_user_count,
        "malicious_ips":     malicious_ip_count,
        "top_malicious_ips": [[ip, cnt] for ip, cnt in top_malicious_ips],
        # Real-time only — last 20 newest first
        "recent_entries":    list(reversed(entries[-20:])),
    }

