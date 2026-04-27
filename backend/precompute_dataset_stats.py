"""
precompute_dataset_stats.py
Run ONCE to compute label distribution and top malicious IPs from reviews_fixed.csv.

Output: backend/dataset_stats.json (used as baseline by the analytics dashboard)

Usage:
    python backend/precompute_dataset_stats.py

Takes ~5-15 min on CPU for 90k rows (TextBlob sentiment is the slow part).
"""

import os
import re
import json
import numpy as np
import pandas as pd
from collections import Counter
from datetime import datetime
from textblob import TextBlob

# ─── Paths ─────────────────────────────────────────────────────────────────────
BACKEND_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR  = os.path.dirname(BACKEND_DIR)
DATASET_PATH = os.path.join(PROJECT_DIR, "reviews_fixed.csv")
STATS_PATH   = os.path.join(BACKEND_DIR, "dataset_stats.json")

# ─── Same weights as colab_train.py ────────────────────────────────────────────
WEIGHTS = {
    'sentiment': 0.15, 'length': 0.05, 'repetition': 0.15,
    'punctuation': 0.10, 'intensifier': 0.10, 'subjectivity': 0.10,
    'unverified': 0.20, 'unhelpful': 0.05, 'ip_freq': 0.10,
    'vine': -0.05,
}

INTENSIFIERS = {
    'amazing', 'awesome', 'terrible', 'horrible', 'worst', 'best', 'perfect',
    'absolutely', 'totally', 'completely', 'extremely', 'incredibly', 'fantastic',
    'outstanding', 'awful', 'disgusting', 'magnificent', 'phenomenal', 'pathetic',
    'superb', 'dreadful', 'flawless', 'useless', 'brilliant', 'rubbish',
    'never', 'always', 'everyone', 'nobody', 'definitely', 'certainly',
}


# ─── Feature helpers ───────────────────────────────────────────────────────────
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s!?.,;\'\"\\-]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def sentiment_extremeness(text):
    return abs(TextBlob(text).sentiment.polarity)


def length_anomaly(text, mean_len, std_len):
    wc = len(text.split())
    z  = abs(wc - mean_len) / (std_len + 1e-6)
    return min(z / 3.0, 1.0)


def repetition_score(text):
    words = text.lower().split()
    if len(words) < 5:
        return 0.0
    wc       = Counter(words)
    repeated = sum(c - 1 for c in wc.values() if c > 1)
    return min(repeated / len(words), 1.0)


def punctuation_abuse(text):
    excl  = text.count('!')
    ques  = text.count('?')
    caps  = sum(1 for w in text.split() if w.isupper() and len(w) > 1)
    total = max(len(text.split()), 1)
    return min((excl + ques + caps * 2) / total, 1.0)


def intensifier_ratio(text):
    words = text.lower().split()
    if not words:
        return 0.0
    count = sum(1 for w in words if w in INTENSIFIERS)
    return min(count / len(words) * 5, 1.0)


def subjectivity_score(text):
    return TextBlob(text).sentiment.subjectivity


# ─── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("Fake Review Detector — Dataset Stats Precomputation")
    print("=" * 60)

    if not os.path.exists(DATASET_PATH):
        print(f"\nERROR: Dataset not found at:\n  {DATASET_PATH}")
        print("Make sure reviews_fixed.csv is in the project root.")
        exit(1)

    # ── Step 1: Load ────────────────────────────────────────────────────────
    print(f"\n[1/5] Loading dataset…")
    df = pd.read_csv(
        DATASET_PATH, on_bad_lines='skip', engine='python',
        quoting=1, encoding='utf-8', encoding_errors='replace'
    )
    cols = [c for c in df.columns if not c.startswith('Unnamed')]
    df   = df[cols]
    print(f"  Raw rows: {len(df):,}")

    # ── Step 2: Prepare ─────────────────────────────────────────────────────
    print("\n[2/5] Preparing columns…")
    df['reviewText'] = (df['review_headline'].fillna('') + " " +
                        df['review_body'].fillna(''))
    df = (df[['reviewText', 'star_rating', 'verified_purchase', 'vine',
              'helpful_votes', 'total_votes', 'IP Address']]
          .dropna(subset=['reviewText'])
          .reset_index(drop=True))

    df['original_text'] = df['reviewText'].astype(str)
    df['clean_text']    = df['original_text'].apply(clean_text)
    df = df[df['clean_text'].str.len() > 10].reset_index(drop=True)
    print(f"  Valid reviews: {len(df):,}")

    # ── Step 3: Compute heuristic features ─────────────────────────────────
    print("\n[3/5] Computing features (TextBlob is the slow part — please wait)…")
    total      = len(df)
    BATCH      = 2000
    sents, subs, lens, reps, puncts, intens = [], [], [], [], [], []

    word_counts      = df['clean_text'].apply(lambda x: len(x.split()))
    mean_len, std_len = float(word_counts.mean()), float(word_counts.std())

    for i in range(0, total, BATCH):
        b_clean = df['clean_text'].iloc[i:i + BATCH]
        b_orig  = df['original_text'].iloc[i:i + BATCH]

        sents.extend(b_clean.apply(sentiment_extremeness).tolist())
        subs.extend(b_clean.apply(subjectivity_score).tolist())
        lens.extend(b_clean.apply(lambda x: length_anomaly(x, mean_len, std_len)).tolist())
        reps.extend(b_clean.apply(repetition_score).tolist())
        puncts.extend(b_orig.apply(punctuation_abuse).tolist())
        intens.extend(b_clean.apply(intensifier_ratio).tolist())

        done = min(i + BATCH, total)
        pct  = done / total * 100
        bar  = '#' * int(pct / 5) + '-' * (20 - int(pct / 5))
        print(f"  [{bar}] {pct:5.1f}%  ({done:,}/{total:,})", end='\r', flush=True)

    print(f"\n  Features computed.")

    df['feat_sentiment']    = sents
    df['feat_subjectivity'] = subs
    df['feat_length']       = lens
    df['feat_repetition']   = reps
    df['feat_punctuation']  = puncts
    df['feat_intensifier']  = intens

    df['feat_unverified'] = (df['verified_purchase'] == 'N').astype(float)
    df['feat_vine']       = (df['vine'] == 'Y').astype(float)

    df['helpful_votes']  = pd.to_numeric(df['helpful_votes'], errors='coerce').fillna(0)
    df['total_votes']    = pd.to_numeric(df['total_votes'],   errors='coerce').fillna(0)
    df['feat_unhelpful'] = np.where(
        df['total_votes'] > 5,
        1.0 - df['helpful_votes'] / df['total_votes'],
        0.0
    )

    ip_counts          = df['IP Address'].value_counts()
    df['feat_ip_freq'] = df['IP Address'].map(ip_counts)
    ip_mean, ip_std    = df['feat_ip_freq'].mean(), df['feat_ip_freq'].std()
    df['feat_ip_freq'] = ((df['feat_ip_freq'] - ip_mean) / (ip_std + 1e-6)).clip(0, 1)

    # ── Step 4: Assign labels ───────────────────────────────────────────────
    print("\n[4/5] Assigning labels…")
    df['manipulation_score_raw'] = (
        WEIGHTS['sentiment']    * df['feat_sentiment']    +
        WEIGHTS['length']       * df['feat_length']       +
        WEIGHTS['repetition']   * df['feat_repetition']   +
        WEIGHTS['punctuation']  * df['feat_punctuation']  +
        WEIGHTS['intensifier']  * df['feat_intensifier']  +
        WEIGHTS['subjectivity'] * df['feat_subjectivity'] +
        WEIGHTS['unverified']   * df['feat_unverified']   +
        WEIGHTS['unhelpful']    * df['feat_unhelpful']    +
        WEIGHTS['ip_freq']      * df['feat_ip_freq']      +
        WEIGHTS['vine']         * df['feat_vine']
    )

    mn = df['manipulation_score_raw'].min()
    mx = df['manipulation_score_raw'].max()
    df['manipulation_score'] = ((df['manipulation_score_raw'] - mn) / (mx - mn)).clip(0, 1)

    t_genuine    = float(df['manipulation_score'].quantile(0.50))
    t_suspicious = float(df['manipulation_score'].quantile(0.80))

    def assign_label(score):
        if score <= t_genuine:      return 'Genuine'
        elif score <= t_suspicious: return 'Suspicious'
        else:                       return 'Manipulative'

    df['label'] = df['manipulation_score'].apply(assign_label)
    label_counts = df['label'].value_counts().to_dict()
    print(f"  Genuine:      {label_counts.get('Genuine', 0):,}")
    print(f"  Suspicious:   {label_counts.get('Suspicious', 0):,}")
    print(f"  Manipulative: {label_counts.get('Manipulative', 0):,}")

    # Top malicious IPs in the dataset
    fake_df = df[df['label'].isin(['Suspicious', 'Manipulative'])]
    top_ips  = [
        [str(ip), int(cnt)]
        for ip, cnt in fake_df['IP Address'].value_counts().head(10).items()
    ]

    # ── Step 5: Save ────────────────────────────────────────────────────────
    print("\n[5/5] Saving dataset_stats.json…")
    stats = {
        "total":             int(len(df)),
        "genuine":           int(label_counts.get('Genuine',      0)),
        "suspicious":        int(label_counts.get('Suspicious',   0)),
        "manipulative":      int(label_counts.get('Manipulative', 0)),
        "top_malicious_ips": top_ips,
        "computed_at":       datetime.now().isoformat(),
    }

    with open(STATS_PATH, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)

    print(f"\n✅ Saved to: {STATS_PATH}")
    print(f"   Total:         {stats['total']:,}")
    print(f"   Genuine:       {stats['genuine']:,}")
    print(f"   Suspicious:    {stats['suspicious']:,}")
    print(f"   Manipulative:  {stats['manipulative']:,}")
    print(f"   Top malicious IPs: {len(top_ips)}")
    print("\nRun the Streamlit dashboard — it will now show dataset-level stats as baseline.")
