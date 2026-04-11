"""
predictor.py — DistilBERT Sequence Classification + GPT-2 Perplexity logic.

Loads fine-tuned DistilBERT models and provides predict_review() function
that returns: prediction (Genuine/Suspicious/Manipulative), confidence, reasons.
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

# ─── Paths ─────────────────────────────────────────────────────────────────────
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
DISTILBERT_DIR = os.path.join(BACKEND_DIR, "distilbert_model")

# ─── Load artifacts at module import time ──────────────────────────────────────
print("[predictor] Loading model artifacts...")

label_encoder = joblib.load(os.path.join(BACKEND_DIR, "label_encoder.pkl"))
thresholds = joblib.load(os.path.join(BACKEND_DIR, "thresholds.pkl"))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[predictor] Using device: {device}")

print("[predictor] Loading DistilBERT model...")
bert_tokenizer = DistilBertTokenizerFast.from_pretrained(DISTILBERT_DIR)
bert_model = DistilBertForSequenceClassification.from_pretrained(DISTILBERT_DIR).to(device)
bert_model.eval()

print("[predictor] Loading GPT-2 model (for Perplexity)...")
gpt2_tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
gpt2_model.eval()

print("[predictor] All models loaded successfully!")


# ─── Text cleaning (same as notebook) ─────────────────────────────────────────
def clean_text(text):
    """Clean review text — same logic as notebook."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'http\S+|www\.\S+', '', str(text))
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s!?.,;\'\"\-]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ─── GPT-2 Perplexity ─────────────────────────────────────────────────────────
def get_perplexity(text, max_length=512):
    """Compute GPT-2 perplexity for a single text."""
    try:
        encodings = gpt2_tokenizer(
            text, return_tensors="pt", truncation=True,
            max_length=max_length
        ).to(device)

        input_ids = encodings["input_ids"]

        if input_ids.size(1) < 2:
            return 100.0

        with torch.no_grad():
            outputs = gpt2_model(input_ids, labels=input_ids)
            loss = outputs.loss

        ppl = torch.exp(loss).item()
        return min(ppl, 10000.0)
    except Exception:
        return 100.0


# ─── Main prediction function ─────────────────────────────────────────────────
def predict_review(review_text, rating, verified, ip, user_id):
    """
    Predict whether a review is Genuine, Suspicious, or Manipulative using DistilBERT.

    Args:
        review_text: str — the review text
        rating: int — star rating (1-5)
        verified: int — 1 if verified purchase, 0 if not
        ip: str — IP address
        user_id: str — user identifier
    """
    # 1. Clean the text explicitly
    cleaned = clean_text(review_text)
    if len(cleaned) < 5:
        return {
            "prediction": "Suspicious",
            "confidence": 0.5,
            "reasons": ["Review text too short to analyze accurately"]
        }

    # 2. Metadata Injection (the exact format the DistilBERT model was trained on)
    verified_str = "Y" if verified == 1 else "N"
    model_text = f"verified {verified_str} rating {float(rating)} review: {cleaned.lower()}"
    
    # 3. GPT-2 Perplexity (calculated purely on the cleaned text itself, no tags)
    perplexity = get_perplexity(cleaned)

    # 4. DistilBERT Prediction
    encodings = bert_tokenizer(
        [model_text], truncation=True, padding='max_length',
        max_length=128, return_tensors='pt'
    )
    
    with torch.no_grad():
        outputs = bert_model(
            input_ids=encodings['input_ids'].to(device),
            attention_mask=encodings['attention_mask'].to(device)
        )
        
    logits = outputs.logits
    probs = F.softmax(logits, dim=-1)[0].cpu().numpy()
    
    pred_idx = np.argmax(probs)
    prediction = label_encoder.inverse_transform([pred_idx])[0]
    confidence = float(probs[pred_idx])

    # Generate original prediction first to pass to reason generator
    original_prediction = prediction
    
    # 5. Generate standard metadata/NLP reasons
    words = cleaned.split()
    word_count = len(words)
    reasons = generate_reasons(
        perplexity, float(rating), verified, word_count, cleaned, review_text, original_prediction
    )

    # 6. Post-Processing Override: Sentiment vs Rating Contradiction
    sentiment = TextBlob(cleaned).sentiment.polarity
    if rating >= 4 and sentiment <= -0.2:
        prediction = "Manipulative"
        confidence = max(confidence, 0.95)
        reasons.append(f"Severe Contradiction: Review text is highly negative but rating is positive ({rating}/5)")
    elif rating <= 2 and sentiment >= 0.5:
        prediction = "Suspicious"
        confidence = max(confidence, 0.85)
        reasons.append(f"Contradiction: Review text is highly positive but rating is negative ({rating}/5)")

    # 7. Check User History tracking
    is_mal, hist = is_malicious_user(user_id)
    if is_mal and hist:
        reasons.append(
            f"⚠️ Malicious user detected: {hist['fake_count']}/{hist['reviews_submitted']} "
            f"past reviews were flagged as fake"
        )
        prediction = "Manipulative"
        confidence = max(confidence, 0.90)  # High confidence if user is explicitly malicious

    # 8. Check IP Integrity
    if verify_ip_integrity(ip, user_id):
        reasons.append("⚠️ IP Integrity Warning: This IP address is associated with multiple distinct user IDs (Sockpuppet behavior)")
        prediction = "Suspicious" if prediction == "Genuine" else "Manipulative"
        confidence = max(confidence, 0.85)

    # 9. Clear negative reasons if it successfully remained Genuine!
    if prediction == "Genuine":
        reasons = ["Semantic patterns match authentic, human-written reviews and verified purchasing behavior."]

    # 10. Update User History tracking if it flagged as fake
    is_fake = prediction in ("Suspicious", "Manipulative")
    update_user_history(user_id, is_fake)

    return {
        "prediction": prediction,
        "confidence": round(confidence, 4),
        "reasons": reasons
    }


def generate_reasons(perplexity, rating, verified, word_count, cleaned, original, original_prediction):
    """Generate human-readable reasons based on explicit statistics tracking."""
    reasons = []

    # Perplexity-based reasoning
    # DistilBERT gives deep semantic scores, but GPT-2 specifically highlights bot/spam behavior
    if perplexity < 30.0:  # Extremely low perplexity often indicates templated/AI text
        reasons.append("Highly templated or generic language (abnormally low perplexity)")
    elif perplexity > 500.0:
        reasons.append("Unusually incoherent or keyword-stuffed language (very high perplexity)")

    # Verified purchase
    if verified == 0:
        reasons.append("Review is not tied to a verified purchase")

    # Rating extremes
    if rating == 5:
        reasons.append("Extreme positive rating (5/5 — potential manipulation signal)")
    elif rating == 1:
        reasons.append("Extreme negative rating (1/5 — potential manipulation signal)")

    # Word count anomalies (Using stats we cached in thresholds.pkl)
    mean_wc = thresholds.get('mean_word_count', 40.0)
    std_wc = thresholds.get('std_word_count', 30.0)
    
    if word_count < 5:
        reasons.append("Very short review (potentially low-effort fake)")
    elif word_count > mean_wc + 2.5 * std_wc:
        reasons.append("Unusually long review compared to platform averages (potential spam)")

    # Punctuation abuse
    excl = original.count('!')
    ques = original.count('?')
    caps_words = sum(1 for w in original.split() if w.isupper() and len(w) > 1)
    total_words = max(word_count, 1)
    punct_score = (excl + ques + caps_words * 2) / total_words
    if punct_score > 0.15:
        reasons.append("Excessive usage of exclamation marks, question marks, or ALL CAPS")

    # Intensifier abuse
    INTENSIFIERS = {
        'amazing', 'awesome', 'terrible', 'horrible', 'worst', 'best', 'perfect',
        'absolutely', 'totally', 'completely', 'extremely', 'incredibly', 'fantastic',
        'outstanding', 'awful', 'disgusting', 'magnificent', 'phenomenal', 'pathetic',
        'superb', 'dreadful', 'flawless', 'useless', 'brilliant', 'rubbish',
        'never', 'always', 'everyone', 'nobody', 'definitely', 'certainly'
    }
    words = cleaned.lower().split()
    if words:
        intens_ratio = sum(1 for w in words if w in INTENSIFIERS) / len(words) * 5
        if intens_ratio > 0.20:
            reasons.append("Abnormally high use of hyperbolic/intensifier language")

    # Repetition
    if len(words) >= 5:
        word_counts = {}
        for w in words:
            word_counts[w] = word_counts.get(w, 0) + 1
        repeated = sum(c - 1 for c in word_counts.values() if c > 1)
        rep_score = repeated / len(words)
        if rep_score > 0.3:
            reasons.append("High amount of repeated phrases detected (spam-like)")

    # If no specific explicit reasons triggered, but the deep learning model flagged it anyway
    if not reasons:
        reasons.append("Pattern matches deep semantic signatures of known fake reviews.")

    return reasons
