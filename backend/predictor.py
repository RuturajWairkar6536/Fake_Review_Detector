"""
predictor.py — DistilBERT Sequence Classification + GPT-2 Perplexity + Advanced Fraud Signals.

Enhanced with:
  1. Temporal burst detection (activity_log module)

Prediction pipeline:
  DistilBERT  →  GPT-2 Perplexity  →  Sentiment/Rating
  →  Temporal Burst  →  User History  →  IP Integrity  →  Final verdict + reasons
"""

import os
import re
import numpy as np
import joblib
import torch
import torch.nn.functional as F
from textblob import TextBlob
from transformers import (DistilBertTokenizerFast, DistilBertForSequenceClassification,
                          GPT2LMHeadModel, GPT2TokenizerFast)
from backend.user_history import is_malicious_user, update_user_history, verify_ip_integrity
from backend.activity_log import log_activity, analyze_user_burst, analyze_ip_burst, detect_midnight_spam

# ─── Paths ─────────────────────────────────────────────────────────────────────
BACKEND_DIR    = os.path.dirname(os.path.abspath(__file__))
DISTILBERT_DIR = os.path.join(BACKEND_DIR, "distilbert_model")

# ─── Load artifacts at module import time ──────────────────────────────────────
print("[predictor] Loading model artifacts...")

label_encoder = joblib.load(os.path.join(BACKEND_DIR, "label_encoder.pkl"))
thresholds    = joblib.load(os.path.join(BACKEND_DIR, "thresholds.pkl"))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[predictor] Using device: {device}")

print("[predictor] Loading DistilBERT model...")
bert_tokenizer = DistilBertTokenizerFast.from_pretrained(DISTILBERT_DIR)
bert_model     = DistilBertForSequenceClassification.from_pretrained(DISTILBERT_DIR).to(device)
bert_model.eval()

print("[predictor] Loading GPT-2 model (for Perplexity)...")
gpt2_tokenizer              = GPT2TokenizerFast.from_pretrained('gpt2')
gpt2_tokenizer.pad_token    = gpt2_tokenizer.eos_token
gpt2_model                  = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
gpt2_model.eval()

print("[predictor] All models loaded successfully!")


# ─── Text Cleaning ─────────────────────────────────────────────────────────────
def clean_text(text):
    """Clean review text — same logic as training notebook."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'http\S+|www\.\S+', '', str(text))
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s!?.,;\'\"\\-]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ─── GPT-2 Perplexity ─────────────────────────────────────────────────────────
def get_perplexity(text, max_length=512):
    """Compute GPT-2 perplexity for a single text string."""
    try:
        encodings = gpt2_tokenizer(
            text, return_tensors="pt", truncation=True, max_length=max_length
        ).to(device)

        input_ids = encodings["input_ids"]
        if input_ids.size(1) < 2:
            return 100.0

        with torch.no_grad():
            outputs = gpt2_model(input_ids, labels=input_ids)
            loss    = outputs.loss

        return min(torch.exp(loss).item(), 10_000.0)
    except Exception:
        return 100.0


# ─── Main Prediction Function ──────────────────────────────────────────────────
def predict_review(review_text, rating, verified, ip, user_id, timestamp=None):
    """
    Predict whether a review is Genuine, Suspicious, or Manipulative.

    Args:
        review_text    : str  — the review body text
        rating         : int  — star rating (1–5)
        verified       : int  — 1 if verified purchase, 0 if not
        ip             : str  — IP address of submitter
        user_id        : str  — user identifier
        timestamp      : str  — (optional) ISO 8601 submission time for burst detection

    Returns:
        {"prediction": str, "confidence": float, "reasons": [str]}
    """
    # 1 ── Clean text
    cleaned = clean_text(review_text)
    if len(cleaned) < 5:
        return {
            "prediction": "Suspicious",
            "confidence": 0.5,
            "reasons": ["Review text too short to analyze accurately"],
        }

    # 2 ── Metadata injection (matches DistilBERT training format exactly)
    verified_str = "Y" if verified == 1 else "N"
    model_text   = f"verified {verified_str} rating {float(rating)} review: {cleaned.lower()}"

    # 3 ── GPT-2 Perplexity
    perplexity = get_perplexity(cleaned)

    # 4 ── DistilBERT Classification
    encodings = bert_tokenizer(
        [model_text], truncation=True, padding='max_length',
        max_length=128, return_tensors='pt'
    )
    with torch.no_grad():
        outputs = bert_model(
            input_ids=encodings['input_ids'].to(device),
            attention_mask=encodings['attention_mask'].to(device),
        )

    logits     = outputs.logits
    probs      = F.softmax(logits, dim=-1)[0].cpu().numpy()
    pred_idx   = int(np.argmax(probs))
    prediction = label_encoder.inverse_transform([pred_idx])[0]
    confidence = float(probs[pred_idx])

    original_prediction = prediction

    # 5 ── Generate base NLP + metadata reasons
    words      = cleaned.split()
    word_count = len(words)
    reasons    = generate_reasons(
        perplexity, float(rating), verified,
        word_count, cleaned, review_text, original_prediction
    )

    # 6 ── Sentiment vs Rating Contradiction
    sentiment = TextBlob(cleaned).sentiment.polarity
    if rating >= 4 and sentiment <= -0.2:
        prediction = "Manipulative"
        confidence = max(confidence, 0.95)
        reasons.append(
            f"Severe Contradiction: Review text is highly negative but rating is positive ({rating}/5)"
        )
    elif rating <= 2 and sentiment >= 0.5:
        prediction = "Suspicious"
        confidence = max(confidence, 0.85)
        reasons.append(
            f"Contradiction: Review text is highly positive but rating is negative ({rating}/5)"
        )

    # 8 ── Temporal Burst Detection (NEW)
    if user_id:
        burst_count, is_burst = analyze_user_burst(user_id, timestamp, window_minutes=5)
        if is_burst:
            reasons.append(
                f"Unusual burst activity: {burst_count} reviews submitted by this user "
                f"within the last 5 minutes."
            )
            prediction = "Suspicious" if prediction == "Genuine" else "Manipulative"
            confidence = max(confidence, 0.87)

        midnight_ratio, is_midnight = detect_midnight_spam(user_id, timestamp)
        if is_midnight:
            reasons.append(
                f"Suspicious posting schedule: {midnight_ratio * 100:.0f}% of this user's "
                f"activity occurs between midnight and 4 AM."
            )
            prediction = prediction if prediction != "Genuine" else "Suspicious"
            confidence = max(confidence, 0.80)

    if ip:
        review_cnt, user_cnt, is_ip_burst = analyze_ip_burst(ip, user_id, timestamp, window_minutes=60)
        if is_ip_burst:
            reasons.append(
                f"Multiple users posted from the same IP in the last hour "
                f"({review_cnt} reviews from {user_cnt} distinct accounts). "
                f"Possible coordinated fraud or bot network."
            )
            prediction = "Manipulative"
            confidence = max(confidence, 0.90)

    # 9 ── User History (existing)
    is_mal, hist = is_malicious_user(user_id)
    if is_mal and hist:
        reasons.append(
            f"⚠️ Malicious user detected: {hist['fake_count']}/{hist['reviews_submitted']} "
            f"past reviews were flagged as fake."
        )
        prediction = "Manipulative"
        confidence = max(confidence, 0.90)

    # 10 ── IP Integrity / Sockpuppet (existing)
    if verify_ip_integrity(ip, user_id):
        reasons.append(
            "⚠️ IP Integrity Warning: This IP address is associated with multiple distinct "
            "user IDs (Sockpuppet behavior)."
        )
        prediction = "Manipulative" if prediction != "Genuine" else "Suspicious"
        confidence = max(confidence, 0.85)

    # 11 ── Clean reasons if still Genuine
    if prediction == "Genuine":
        reasons = [
            "Semantic patterns match authentic, human-written reviews "
            "and verified purchasing behavior."
        ]

    # 12 ── Update Persistent User History
    is_fake = prediction in ("Suspicious", "Manipulative")
    update_user_history(user_id, is_fake)

    # 13 ── Log Activity for Temporal Analysis (NEW)
    log_activity(user_id, ip, timestamp, prediction)

    return {
        "prediction": prediction,
        "confidence": round(confidence, 4),
        "reasons":    reasons,
    }


# ─── Reason Generation ────────────────────────────────────────────────────────
def generate_reasons(perplexity, rating, verified, word_count, cleaned, original,
                     original_prediction):
    """Generate human-readable fraud signals from NLP heuristics and metadata."""
    reasons = []

    # GPT-2 Perplexity
    if perplexity < 30.0:
        reasons.append("Highly templated or generic language (abnormally low perplexity).")
    elif perplexity > 500.0:
        reasons.append("Unusually incoherent or keyword-stuffed language (very high perplexity).")

    # Purchase verification
    if verified == 0:
        reasons.append("Review is not tied to a verified purchase.")

    # Rating extremes
    # Removed as 1 and 5 star ratings are common

    # Word count anomalies
    mean_wc = thresholds.get('mean_word_count', 40.0)
    std_wc  = thresholds.get('std_word_count', 30.0)
    if word_count < 5:
        reasons.append("Very short review (potentially low-effort fake).")
    elif word_count > mean_wc + 2.5 * std_wc:
        reasons.append("Unusually long review compared to platform averages (potential spam).")

    # Punctuation abuse
    excl       = original.count('!')
    ques       = original.count('?')
    caps_words = sum(1 for w in original.split() if w.isupper() and len(w) > 1)
    total_w    = max(word_count, 1)
    if (excl + ques + caps_words * 2) / total_w > 0.15:
        reasons.append("Excessive usage of exclamation marks, question marks, or ALL CAPS.")

    # Intensifier overuse
    INTENSIFIERS = {
        'amazing', 'awesome', 'terrible', 'horrible', 'worst', 'best', 'perfect',
        'absolutely', 'totally', 'completely', 'extremely', 'incredibly', 'fantastic',
        'outstanding', 'awful', 'disgusting', 'magnificent', 'phenomenal', 'pathetic',
        'superb', 'dreadful', 'flawless', 'useless', 'brilliant', 'rubbish',
        'never', 'always', 'everyone', 'nobody', 'definitely', 'certainly',
    }
    words_lower = cleaned.lower().split()
    if words_lower:
        intens_ratio = sum(1 for w in words_lower if w in INTENSIFIERS) / len(words_lower) * 5
        if intens_ratio > 0.20:
            reasons.append("Abnormally high use of hyperbolic/intensifier language.")

    # Word repetition
    if len(words_lower) >= 5:
        wc_map   = {}
        for w in words_lower:
            wc_map[w] = wc_map.get(w, 0) + 1
        repeated  = sum(c - 1 for c in wc_map.values() if c > 1)
        rep_score = repeated / len(words_lower)
        if rep_score > 0.3:
            reasons.append("High amount of repeated phrases detected (spam-like).")

    # Fallback: DistilBERT flagged it semantically
    if not reasons:
        reasons.append(
            "Pattern matches deep semantic signatures of known fake reviews."
        )

    return reasons
