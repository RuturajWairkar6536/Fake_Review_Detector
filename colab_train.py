import os
import re
import warnings
import time
import numpy as np
import pandas as pd
import joblib
from collections import Counter
from textblob import TextBlob
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import torch
from sentence_transformers import SentenceTransformer
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

warnings.filterwarnings("ignore")

# ─── Configuration ─────────────────────────────────────────────────────────────
# Mount Google Drive to save artifacts
from google.colab import drive
drive.mount('/content/drive')

# Upload your reviews_fixed.csv to the Colab session first
DATASET_PATH = "/content/reviews_fixed.csv"
# We will save models to your Google Drive to easily download them later
OUTPUT_DIR = "/content/drive/MyDrive/fake_review_models"
os.makedirs(OUTPUT_DIR, exist_ok=True)

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
    if not isinstance(text, str): return ""
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s!?.,;\'\"\-]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def sentiment_extremeness(text): return abs(TextBlob(text).sentiment.polarity)

def length_anomaly(text, mean_len, std_len):
    word_count = len(text.split())
    z_score = abs(word_count - mean_len) / (std_len + 1e-6)
    return min(z_score / 3.0, 1.0)

def repetition_score(text):
    words = text.lower().split()
    if len(words) < 5: return 0.0
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
    if not words: return 0.0
    count = sum(1 for w in words if w in INTENSIFIERS)
    return min(count / len(words) * 5, 1.0)

def subjectivity_score(text): return TextBlob(text).sentiment.subjectivity

def compute_perplexity_batch(texts, tokenizer, model, device, max_length=512):
    perplexities = []
    # Make sure text is list of strings
    texts = [str(x) if x is not None else "" for x in texts]
    try:
        encodings = tokenizer(
            texts, return_tensors="pt", truncation=True, padding=True,
            max_length=max_length
        ).to(device)
        
        input_ids = encodings["input_ids"]
        attention_mask = encodings["attention_mask"]
        
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
            # We want token-wise loss, so shift labels
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=tokenizer.pad_token_id)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss = loss.view(shift_labels.size())
            
            # Average loss per sequence
            seq_lens = attention_mask[..., 1:].sum(dim=1)
            # Avoid division by zero
            seq_lens = torch.where(seq_lens == 0, torch.tensor(1.0).to(device), seq_lens)
            avg_loss = loss.sum(dim=1) / seq_lens
            ppl = torch.exp(avg_loss)
            
            # Cap at 10000
            ppl = torch.clamp(ppl, max=10000.0)
            perplexities.extend(ppl.cpu().tolist())
    except Exception as e:
        print(f"Error computing perplexity: {e}")
        perplexities.extend([100.0] * len(texts))
    return perplexities

# ─── Training Pipeline ──────────────────────────────────────────────────────────
print("=" * 60)
print("FAKE REVIEW DETECTOR — TPU/GPU BATCH TRAINING")
print("=" * 60)

# Step 1: Load Data
print("\n[1/8] Loading dataset...")
df = pd.read_csv(
    DATASET_PATH, on_bad_lines='skip', engine='python', 
    quoting=1, encoding='utf-8', encoding_errors='replace'
)
cols_to_keep = [c for c in df.columns if not c.startswith('Unnamed')]
df = df[cols_to_keep]

print("\n[2/8] Preparing data...")
df['reviewText'] = df['review_headline'].fillna('') + " " + df['review_body'].fillna('')
df['overall'] = df['star_rating']
df = df[['reviewText', 'overall', 'verified_purchase', 'vine',
         'helpful_votes', 'total_votes', 'IP Address']].dropna(subset=['reviewText']).reset_index(drop=True)

df['original_text'] = df['reviewText'].astype(str)
df['clean_text'] = df['original_text'].apply(clean_text)
df = df[df['clean_text'].str.len() > 10].reset_index(drop=True)
print(f"  Total valid reviews: {len(df)}")

print("\n[3/8] Extracting features & labels...")
word_counts = df['clean_text'].apply(lambda x: len(x.split()))
mean_len, std_len = word_counts.mean(), word_counts.std()

df['feat_sentiment'] = df['clean_text'].apply(sentiment_extremeness)
df['feat_length'] = df['clean_text'].apply(lambda x: length_anomaly(x, mean_len, std_len))
df['feat_repetition'] = df['clean_text'].apply(repetition_score)
df['feat_punctuation'] = df['original_text'].apply(punctuation_abuse)
df['feat_intensifier'] = df['clean_text'].apply(intensifier_ratio)
df['feat_subjectivity'] = df['clean_text'].apply(subjectivity_score)

df['feat_unverified'] = (df['verified_purchase'] == 'N').astype(float)
df['feat_vine'] = (df['vine'] == 'Y').astype(float)
df['helpful_votes'] = pd.to_numeric(df['helpful_votes'], errors='coerce').fillna(0)
df['total_votes'] = pd.to_numeric(df['total_votes'], errors='coerce').fillna(0)
df['feat_unhelpful'] = np.where(df['total_votes'] > 5, 1.0 - (df['helpful_votes'] / df['total_votes']), 0)

ip_counts = df['IP Address'].value_counts()
df['feat_ip_freq'] = df['IP Address'].map(ip_counts)
ip_mean, ip_std = df['feat_ip_freq'].mean(), df['feat_ip_freq'].std()
df['feat_ip_freq'] = ((df['feat_ip_freq'] - ip_mean) / (ip_std + 1e-6)).clip(0, 1)

df['manipulation_score_raw'] = (
    WEIGHTS['sentiment'] * df['feat_sentiment'] + WEIGHTS['length'] * df['feat_length'] +
    WEIGHTS['repetition'] * df['feat_repetition'] + WEIGHTS['punctuation'] * df['feat_punctuation'] +
    WEIGHTS['intensifier'] * df['feat_intensifier'] + WEIGHTS['subjectivity'] * df['feat_subjectivity'] +
    WEIGHTS['unverified'] * df['feat_unverified'] + WEIGHTS['unhelpful'] * df['feat_unhelpful'] +
    WEIGHTS['ip_freq'] * df['feat_ip_freq'] + WEIGHTS['vine'] * df['feat_vine']
)

min_score, max_score = df['manipulation_score_raw'].min(), df['manipulation_score_raw'].max()
df['manipulation_score'] = ((df['manipulation_score_raw'] - min_score) / (max_score - min_score)).clip(0, 1)

threshold_genuine = df['manipulation_score'].quantile(0.50)
threshold_suspicious = df['manipulation_score'].quantile(0.80)

def assign_label(score):
    if score <= threshold_genuine: return 'Genuine'
    elif score <= threshold_suspicious: return 'Suspicious'
    else: return 'Manipulative'

df['label'] = df['manipulation_score'].apply(assign_label)

# Since you want to train on 90k, let's use the full df (no sampling)
# However, to be memory safe on Colab, we can do it in batches during inference.
print(f"  Training on {len(df)} reviews")

print("\n[5/8] Computing SBERT embeddings (GPU)...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sbert = SentenceTransformer('all-MiniLM-L6-v2', device=device)
texts = df['clean_text'].tolist()

start_time = time.time()
# Batch size for SBERT on T4 can be quite large
embeddings = sbert.encode(texts, show_progress_bar=True, batch_size=256)
print(f"  SBERT done in {time.time() - start_time:.1f}s")

print("\n[6/8] Computing GPT-2 perplexity (GPU Batching)...")
gpt2_tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
gpt2_model.eval()

start_time = time.time()
batch_size = 32  # Increase batch size for T4
perplexities = []
total = len(texts)

for i in range(0, total, batch_size):
    batch = texts[i:i + batch_size]
    ppl = compute_perplexity_batch(batch, gpt2_tokenizer, gpt2_model, device)
    perplexities.extend(ppl)
    if (i + 1) % 1000 == 0 or i + batch_size >= total:
        print(f"  Processed {min(i + batch_size, total)}/{total}")

print(f"  GPT-2 perplexity done in {time.time() - start_time:.1f}s")
df['perplexity'] = perplexities

ppl_array = np.array(perplexities)
ppl_25, ppl_75 = float(np.percentile(ppl_array, 25)), float(np.percentile(ppl_array, 75))

print("\n[7/8] Sub-sampling features & training XGBoost...")
df['rating_numeric'] = pd.to_numeric(df['overall'], errors='coerce').fillna(3).astype(float)
df['verified_numeric'] = (df['verified_purchase'] == 'Y').astype(float)
df['word_count'] = df['clean_text'].apply(lambda x: len(x.split()))

metadata_features = df[['perplexity', 'rating_numeric', 'verified_numeric', 'word_count', 'feat_ip_freq']].values
X = np.hstack([embeddings, metadata_features])

le = LabelEncoder()
y = le.fit_transform(df['label'])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

sample_weights = compute_sample_weight('balanced', y_train)
xgb_model = XGBClassifier(
    n_estimators=150,  # More estimators for higher accuracy
    max_depth=6,       # shallower trees for XGB
    learning_rate=0.1,
    tree_method='hist', # Extremely fast histogram method
    random_state=42,
    n_jobs=-1
)
start_time = time.time()
xgb_model.fit(X_train, y_train, sample_weight=sample_weights)
print(f"  XGBoost training done in {time.time() - start_time:.1f}s")

y_pred = xgb_model.predict(X_test)
print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred, target_names=le.classes_))

print("\n[8/8] Saving artifacts...")
joblib.dump(xgb_model, os.path.join(OUTPUT_DIR, "model.pkl"))
joblib.dump(scaler, os.path.join(OUTPUT_DIR, "scaler.pkl"))
joblib.dump(le, os.path.join(OUTPUT_DIR, "label_encoder.pkl"))

thresholds = {
    'perplexity_mean': float(np.mean(ppl_array)),
    'perplexity_std': float(np.std(ppl_array)),
    'perplexity_25': ppl_25, 'perplexity_75': ppl_75,
    'mean_word_count': float(mean_len), 'std_word_count': float(std_len),
    'ip_mean': float(ip_mean), 'ip_std': float(ip_std),
    'threshold_genuine': float(threshold_genuine),
    'threshold_suspicious': float(threshold_suspicious),
    'sbert_model': 'all-MiniLM-L6-v2',
}
joblib.dump(thresholds, os.path.join(OUTPUT_DIR, "thresholds.pkl"))

print(f"🎉 Saved to Google Drive -> {OUTPUT_DIR}/")
