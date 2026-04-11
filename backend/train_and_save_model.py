"""
train_and_save_model.py — Trains the Fake Review Detection model and saves artifacts.

Pipeline:
1. Load reviews_fixed.csv from NLP_MP folder
2. Clean text (same as notebook)
3. Generate manipulation labels using notebook's heuristic pipeline
4. Compute SBERT embeddings (all-MiniLM-L6-v2)
5. Compute GPT-2 perplexity scores
6. Extract metadata features (rating, verified, word_count, ip_frequency)
7. Train Random Forest classifier (3-class: Genuine / Suspicious / Manipulative)
8. Save: model.pkl, scaler.pkl, thresholds.pkl
"""

import os
import sys
import re
import warnings
import time

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
from collections import Counter

from textblob import TextBlob
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# ─── Paths ─────────────────────────────────────────────────────────────────────
DATASET_PATH = os.path.join(os.path.expanduser("~"), "Desktop", "NLP_MP", "reviews_fixed.csv")
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

SAMPLE_SIZE = 10000  # Use 10K sample for practical GPT-2 perplexity computation

# ─── Notebook's exact feature functions ─────────────────────────────────────────

INTENSIFIERS = {
    'amazing', 'awesome', 'terrible', 'horrible', 'worst', 'best', 'perfect',
    'absolutely', 'totally', 'completely', 'extremely', 'incredibly', 'fantastic',
    'outstanding', 'awful', 'disgusting', 'magnificent', 'phenomenal', 'pathetic',
    'superb', 'dreadful', 'flawless', 'useless', 'brilliant', 'rubbish',
    'never', 'always', 'everyone', 'nobody', 'definitely', 'certainly'
}

WEIGHTS = {
    'sentiment': 0.15, 'length': 0.05, 'repetition': 0.15,
    'punctuation': 0.10, 'intensifier': 0.10, 'subjectivity': 0.10,
    'unverified': 0.20, 'unhelpful': 0.05, 'ip_freq': 0.10,
    'vine': -0.05
}


def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s!?.,;\'\"\-]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def sentiment_extremeness(text):
    return abs(TextBlob(text).sentiment.polarity)


def length_anomaly(text, mean_len, std_len):
    word_count = len(text.split())
    z_score = abs(word_count - mean_len) / (std_len + 1e-6)
    return min(z_score / 3.0, 1.0)


def repetition_score(text):
    words = text.lower().split()
    if len(words) < 5:
        return 0.0
    word_counts = Counter(words)
    repeated = sum(c - 1 for c in word_counts.values() if c > 1)
    return min(repeated / len(words), 1.0)


def punctuation_abuse(text):
    excl, ques = text.count('!'), text.count('?')
    caps_words = sum(1 for w in text.split() if w.isupper() and len(w) > 1)
    total_words = max(len(text.split()), 1)
    return min((excl + ques + caps_words * 2) / total_words, 1.0)


def intensifier_ratio(text):
    words = text.lower().split()
    if not words:
        return 0.0
    count = sum(1 for w in words if w in INTENSIFIERS)
    return min(count / len(words) * 5, 1.0)


def subjectivity_score(text):
    return TextBlob(text).sentiment.subjectivity


def compute_perplexity_batch(texts, tokenizer, model, device, max_length=512):
    """Compute GPT-2 perplexity for a batch of texts."""
    import torch

    perplexities = []
    for text in texts:
        try:
            encodings = tokenizer(
                text, return_tensors="pt", truncation=True,
                max_length=max_length
            ).to(device)
            
            input_ids = encodings["input_ids"]
            
            if input_ids.size(1) < 2:
                perplexities.append(100.0)  # default for very short text
                continue
            
            with torch.no_grad():
                outputs = model(input_ids, labels=input_ids)
                loss = outputs.loss
            
            ppl = torch.exp(loss).item()
            # Cap at reasonable value
            ppl = min(ppl, 10000.0)
            perplexities.append(ppl)
        except Exception:
            perplexities.append(100.0)
    
    return perplexities


def main():
    print("=" * 60)
    print("FAKE REVIEW DETECTOR — MODEL TRAINING")
    print("=" * 60)

    # ─── Step 1: Load Dataset ───────────────────────────────────────────────────
    print("\n[1/8] Loading dataset...")
    df = pd.read_csv(
        DATASET_PATH,
        on_bad_lines='skip',
        engine='python',
        quoting=1,
        encoding='utf-8',
        encoding_errors='replace'
    )

    # Drop Unnamed columns
    cols_to_keep = [c for c in df.columns if not c.startswith('Unnamed')]
    df = df[cols_to_keep]
    print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")

    # ─── Step 2: Prepare data (same as notebook) ───────────────────────────────
    print("\n[2/8] Preparing data...")

    # Merge headline + body
    df['reviewText'] = df['review_headline'].fillna('') + " " + df['review_body'].fillna('')
    df['overall'] = df['star_rating']

    df = df[['reviewText', 'overall', 'verified_purchase', 'vine',
             'helpful_votes', 'total_votes', 'IP Address']].dropna(subset=['reviewText']).reset_index(drop=True)

    # Clean text
    df['original_text'] = df['reviewText'].astype(str)
    df['clean_text'] = df['original_text'].apply(clean_text)
    df = df[df['clean_text'].str.len() > 10].reset_index(drop=True)
    print(f"  After cleaning: {len(df)} reviews")

    # ─── Step 3: Extract linguistic features & compute labels ──────────────────
    print("\n[3/8] Extracting linguistic features & computing manipulation labels...")

    word_counts = df['clean_text'].apply(lambda x: len(x.split()))
    mean_len = word_counts.mean()
    std_len = word_counts.std()

    df['feat_sentiment'] = df['clean_text'].apply(sentiment_extremeness)
    df['feat_length'] = df['clean_text'].apply(lambda x: length_anomaly(x, mean_len, std_len))
    df['feat_repetition'] = df['clean_text'].apply(repetition_score)
    df['feat_punctuation'] = df['original_text'].apply(punctuation_abuse)
    df['feat_intensifier'] = df['clean_text'].apply(intensifier_ratio)
    df['feat_subjectivity'] = df['clean_text'].apply(subjectivity_score)

    # Metadata features
    df['feat_unverified'] = (df['verified_purchase'] == 'N').astype(float)
    df['feat_vine'] = (df['vine'] == 'Y').astype(float)
    df['helpful_votes'] = pd.to_numeric(df['helpful_votes'], errors='coerce').fillna(0)
    df['total_votes'] = pd.to_numeric(df['total_votes'], errors='coerce').fillna(0)
    df['feat_unhelpful'] = np.where(
        df['total_votes'] > 5,
        1.0 - (df['helpful_votes'] / df['total_votes']),
        0
    )

    # IP frequency
    ip_counts = df['IP Address'].value_counts()
    df['feat_ip_freq'] = df['IP Address'].map(ip_counts)
    ip_mean = df['feat_ip_freq'].mean()
    ip_std = df['feat_ip_freq'].std()
    df['feat_ip_freq'] = ((df['feat_ip_freq'] - ip_mean) / (ip_std + 1e-6)).clip(0, 1)

    # Manipulation score (same as notebook)
    df['manipulation_score_raw'] = (
        WEIGHTS['sentiment'] * df['feat_sentiment'] +
        WEIGHTS['length'] * df['feat_length'] +
        WEIGHTS['repetition'] * df['feat_repetition'] +
        WEIGHTS['punctuation'] * df['feat_punctuation'] +
        WEIGHTS['intensifier'] * df['feat_intensifier'] +
        WEIGHTS['subjectivity'] * df['feat_subjectivity'] +
        WEIGHTS['unverified'] * df['feat_unverified'] +
        WEIGHTS['unhelpful'] * df['feat_unhelpful'] +
        WEIGHTS['ip_freq'] * df['feat_ip_freq'] +
        WEIGHTS['vine'] * df['feat_vine']
    )

    min_score = df['manipulation_score_raw'].min()
    max_score = df['manipulation_score_raw'].max()
    df['manipulation_score'] = ((df['manipulation_score_raw'] - min_score) / (max_score - min_score)).clip(0, 1)

    threshold_genuine = df['manipulation_score'].quantile(0.50)
    threshold_suspicious = df['manipulation_score'].quantile(0.80)

    def assign_label(score):
        if score <= threshold_genuine:
            return 'Genuine'
        elif score <= threshold_suspicious:
            return 'Suspicious'
        else:
            return 'Manipulative'

    df['label'] = df['manipulation_score'].apply(assign_label)
    print(f"  Label distribution:\n{df['label'].value_counts().to_string()}")

    # ─── Step 4: Sample for training ───────────────────────────────────────────
    print(f"\n[4/8] Sampling {SAMPLE_SIZE} reviews for training...")

    # Stratified sample
    df_sample = df.groupby('label', group_keys=False).apply(
        lambda x: x.sample(n=min(len(x), SAMPLE_SIZE // 3), random_state=42)
    ).reset_index(drop=True)
    print(f"  Sample size: {len(df_sample)}")
    print(f"  Sample label distribution:\n{df_sample['label'].value_counts().to_string()}")

    # ─── Step 5: SBERT Embeddings ──────────────────────────────────────────────
    print("\n[5/8] Computing SBERT embeddings (all-MiniLM-L6-v2)...")
    from sentence_transformers import SentenceTransformer

    sbert = SentenceTransformer('all-MiniLM-L6-v2')
    texts = df_sample['clean_text'].tolist()
    
    start_time = time.time()
    embeddings = sbert.encode(texts, show_progress_bar=True, batch_size=64)
    elapsed = time.time() - start_time
    print(f"  SBERT embeddings done in {elapsed:.1f}s — shape: {embeddings.shape}")

    # ─── Step 6: GPT-2 Perplexity ─────────────────────────────────────────────
    print("\n[6/8] Computing GPT-2 perplexity scores...")
    import torch
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Using device: {device}")

    gpt2_tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
    gpt2_model.eval()

    start_time = time.time()
    batch_size = 1  # GPT-2 perplexity computed one at a time
    perplexities = []
    total = len(texts)

    for i in range(0, total, batch_size):
        batch = texts[i:i + batch_size]
        ppl = compute_perplexity_batch(batch, gpt2_tokenizer, gpt2_model, device)
        perplexities.extend(ppl)
        if (i + 1) % 500 == 0 or i + batch_size >= total:
            print(f"  Perplexity: {min(i + batch_size, total)}/{total} done")

    elapsed = time.time() - start_time
    print(f"  GPT-2 perplexity done in {elapsed:.1f}s")

    df_sample['perplexity'] = perplexities

    # Save perplexity stats for reasoning thresholds
    ppl_array = np.array(perplexities)
    ppl_mean = float(np.mean(ppl_array))
    ppl_std = float(np.std(ppl_array))
    ppl_25 = float(np.percentile(ppl_array, 25))
    ppl_75 = float(np.percentile(ppl_array, 75))

    # ─── Step 7: Build feature matrix ──────────────────────────────────────────
    print("\n[7/8] Building feature matrix and training Random Forest...")

    # Convert rating to numeric
    df_sample['rating_numeric'] = pd.to_numeric(df_sample['overall'], errors='coerce').fillna(3).astype(float)
    
    # Verified purchase to numeric
    df_sample['verified_numeric'] = (df_sample['verified_purchase'] == 'Y').astype(float)

    # Word count
    df_sample['word_count'] = df_sample['clean_text'].apply(lambda x: len(x.split()))

    # IP frequency (already computed as feat_ip_freq)
    
    # Build feature matrix: SBERT(384) + perplexity + rating + verified + word_count + ip_freq
    metadata_features = df_sample[['perplexity', 'rating_numeric', 'verified_numeric',
                                    'word_count', 'feat_ip_freq']].values
    
    X = np.hstack([embeddings, metadata_features])
    print(f"  Feature matrix shape: {X.shape}")

    # Labels
    le = LabelEncoder()
    y = le.fit_transform(df_sample['label'])
    print(f"  Classes: {list(le.classes_)}")

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train: {len(X_train)} | Test: {len(X_test)}")

    # Train Random Forest
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)

    # Evaluate
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n  Accuracy: {accuracy:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=le.classes_)}")

    # ─── Step 8: Save artifacts ────────────────────────────────────────────────
    print("\n[8/8] Saving model artifacts...")

    # Save model
    joblib.dump(rf, os.path.join(OUTPUT_DIR, "model.pkl"))
    print(f"  ✅ Saved model.pkl")

    # Save scaler
    joblib.dump(scaler, os.path.join(OUTPUT_DIR, "scaler.pkl"))
    print(f"  ✅ Saved scaler.pkl")

    # Save label encoder
    joblib.dump(le, os.path.join(OUTPUT_DIR, "label_encoder.pkl"))
    print(f"  ✅ Saved label_encoder.pkl")

    # Save thresholds and stats for reasoning/inference
    thresholds = {
        'perplexity_mean': ppl_mean,
        'perplexity_std': ppl_std,
        'perplexity_25': ppl_25,
        'perplexity_75': ppl_75,
        'mean_word_count': float(mean_len),
        'std_word_count': float(std_len),
        'ip_mean': float(ip_mean),
        'ip_std': float(ip_std),
        'threshold_genuine': float(threshold_genuine),
        'threshold_suspicious': float(threshold_suspicious),
        'sbert_model': 'all-MiniLM-L6-v2',
    }
    joblib.dump(thresholds, os.path.join(OUTPUT_DIR, "thresholds.pkl"))
    print(f"  ✅ Saved thresholds.pkl")

    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE! All artifacts saved.")
    print("=" * 60)


if __name__ == "__main__":
    main()
