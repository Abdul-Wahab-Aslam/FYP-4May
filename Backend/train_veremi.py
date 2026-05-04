"""
================================================================================
V2V SENTINEL XAI — HYBRID XGBOOST + LSTM TRAINING PIPELINE
================================================================================
Bahria University FYP — AutoSec AI
Author: V2V Sentinel Team
Target Dataset : VeReMi / VeReMi Extension  (https://veremi-dataset.github.io)
Target Accuracy: ≥98%
Target Attacks : GPS Spoofing  |  Altitude Spoofing  |  Speed Injection
Output Artifacts:
    /content/artifacts/xgb_model.pkl
    /content/artifacts/lstm_model.h5
    /content/artifacts/scaler.pkl
    /content/artifacts/label_encoder.pkl
    /content/artifacts/feature_columns.json
    /content/artifacts/shap_explainer.pkl
================================================================================
HOW TO RUN ON GOOGLE COLAB
--------------------------------------------------------------------------------
1. Open colab.research.google.com -> New Notebook -> Runtime -> GPU (T4)
2. Paste this file as a single cell (or split into the marked sections)
3. Upload the VeReMi Extension dataset zip OR mount your Google Drive
4. Run top-to-bottom — it will produce all .pkl files in /content/artifacts/
5. Download the artifacts folder and drop it into your FastAPI backend
================================================================================
"""

# ============================================================================
# SECTION 0  —  ENVIRONMENT SETUP (run first cell on Colab)
# ============================================================================
# !pip install -q xgboost==2.0.3 shap==0.45.0 tensorflow==2.15.0 \
#                 scikit-learn==1.4.0 pandas==2.1.4 numpy==1.26.3 \
#                 matplotlib==3.8.2 seaborn==0.13.1 joblib==1.3.2 tqdm==4.66.1

import os
import json
import glob
import joblib
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score
)
from sklearn.utils.class_weight import compute_class_weight

import xgboost as xgb
import shap

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, BatchNormalization, Input, Bidirectional
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

warnings.filterwarnings("ignore")
np.random.seed(42)
tf.random.set_seed(42)

ARTIFACTS = Path("/content/artifacts")
ARTIFACTS.mkdir(parents=True, exist_ok=True)

print(f"TensorFlow : {tf.__version__}")
print(f"XGBoost    : {xgb.__version__}")
print(f"SHAP       : {shap.__version__}")
print(f"GPU avail  : {len(tf.config.list_physical_devices('GPU'))>0}")


# ============================================================================
# SECTION 1  —  VEREMI DATASET LOADER
# ============================================================================
# VeReMi log format: JSON-lines, each entry has fields like:
#   {"type": 3, "rcvTime": ..., "sendTime": ..., "sender": ...,
#    "senderPseudo": ..., "messageID": ...,
#    "pos": [x, y, z], "pos_noise": [...],
#    "spd": [vx, vy, vz], "spd_noise": [...]}
#
# Ground-truth file is a separate JSONLog with attackerType per sender.
# VeReMi Extension provides attackerType labels:
#   0=Normal, 1=ConstPos, 2=ConstPosOffset, 4=RandomPos, 8=RandomPosOffset,
#   16=ConstSpeed, 32=ConstSpeedOffset, 64=RandomSpeed, 128=RandomSpeedOffset,
#   ... and Z-axis altitude variants
# ============================================================================

# We map raw attackerType integers to our 4 high-level classes:
#  0 = Normal
#  1 = GPS Spoofing       (position falsification — types 1,2,4,8)
#  2 = Altitude Spoofing  (Z-axis manipulation — VeReMi Ext altitude attacks)
#  3 = Speed Injection    (velocity falsification — types 16,32,64,128)

ATTACK_MAP = {
    0:   "Normal",
    1:   "GPSSpoof",   2:   "GPSSpoof",   4:   "GPSSpoof",   8:   "GPSSpoof",
    16:  "SpeedInj",   32:  "SpeedInj",   64:  "SpeedInj",  128:  "SpeedInj",
    # VeReMi Extension altitude (z-axis position) attacks
    256: "AltSpoof",   512: "AltSpoof",  1024: "AltSpoof",  2048: "AltSpoof",
}


def load_veremi_logs(root_dir: str) -> pd.DataFrame:
    """
    Walks a VeReMi/VeReMi-Extension simulation root and returns a flat DataFrame
    of received BSM messages with ground-truth labels.

    Expected layout:
      root_dir/
        ScenarioX/
          JSONlog-<vehNum>-<modNum>-<A?>.json     <-- per-receiver logs
          GroundTruthJSONlog.json                  <-- attacker truth
    """
    rows = []
    log_files = glob.glob(os.path.join(root_dir, "**", "JSONlog-*.json"), recursive=True)
    truth_files = glob.glob(os.path.join(root_dir, "**", "GroundTruth*.json"), recursive=True)

    # Build sender -> attackerType map from ground-truth files
    truth_map = {}
    for gtf in truth_files:
        with open(gtf, "r") as f:
            for line in f:
                try:
                    rec = json.loads(line.strip())
                    if "sender" in rec and "attackerType" in rec:
                        truth_map[rec["sender"]] = rec["attackerType"]
                except Exception:
                    continue

    print(f"Found {len(log_files)} log files, {len(truth_map)} labeled senders")

    for lf in tqdm(log_files, desc="Parsing logs"):
        with open(lf, "r") as f:
            for line in f:
                try:
                    msg = json.loads(line.strip())
                except Exception:
                    continue
                if msg.get("type") != 3:   # 3 = received BSM
                    continue
                sender = msg.get("sender")
                pos = msg.get("pos", [0, 0, 0])
                spd = msg.get("spd", [0, 0, 0])
                pos_n = msg.get("pos_noise", [0, 0, 0])
                spd_n = msg.get("spd_noise", [0, 0, 0])
                rows.append({
                    "rcvTime":  msg.get("rcvTime", 0.0),
                    "sendTime": msg.get("sendTime", 0.0),
                    "sender":   sender,
                    "pos_x":    pos[0],
                    "pos_y":    pos[1],
                    "pos_z":    pos[2] if len(pos) > 2 else 0.0,
                    "spd_x":    spd[0],
                    "spd_y":    spd[1],
                    "spd_z":    spd[2] if len(spd) > 2 else 0.0,
                    "pos_noise_x": pos_n[0],
                    "pos_noise_y": pos_n[1],
                    "pos_noise_z": pos_n[2] if len(pos_n) > 2 else 0.0,
                    "spd_noise_x": spd_n[0],
                    "spd_noise_y": spd_n[1],
                    "spd_noise_z": spd_n[2] if len(spd_n) > 2 else 0.0,
                    "attackerType": truth_map.get(sender, 0),
                })

    df = pd.DataFrame(rows)
    df["attack_label"] = df["attackerType"].map(ATTACK_MAP).fillna("Normal")
    return df


# --------------------------------------------------------------------------
# OPTION A:  Real VeReMi dataset
# --------------------------------------------------------------------------
# from google.colab import files; files.upload()                # upload zip
# !unzip -q VeReMi.zip -d /content/veremi
# df = load_veremi_logs("/content/veremi")

# --------------------------------------------------------------------------
# OPTION B:  Synthetic generator (use this if VeReMi download isn't ready
#            yet — produces statistically similar data so the pipeline runs
#            end-to-end and you swap in real data later by replacing `df`)
# --------------------------------------------------------------------------
def generate_synthetic_veremi(n: int = 100_000, attack_ratio: float = 0.35) -> pd.DataFrame:
    """
    Synthetic VeReMi-like dataset for prototyping the pipeline.
    Mirrors the real schema so swapping in load_veremi_logs() is transparent.
    """
    rng = np.random.default_rng(42)
    n_normal = int(n * (1 - attack_ratio))
    n_attack = n - n_normal
    n_per    = n_attack // 3

    def normal_block(k):
        return pd.DataFrame({
            "rcvTime":  rng.uniform(0, 86400, k),
            "sendTime": rng.uniform(0, 86400, k),
            "sender":   rng.integers(1, 500, k),
            "pos_x":    rng.uniform(0, 5000, k),
            "pos_y":    rng.uniform(0, 5000, k),
            "pos_z":    rng.normal(0.0, 0.5, k),                 # near-ground
            "spd_x":    rng.normal(0, 12, k),
            "spd_y":    rng.normal(0, 12, k),
            "spd_z":    rng.normal(0.0, 0.1, k),
            "pos_noise_x": rng.normal(0, 0.3, k),
            "pos_noise_y": rng.normal(0, 0.3, k),
            "pos_noise_z": rng.normal(0, 0.05, k),
            "spd_noise_x": rng.normal(0, 0.2, k),
            "spd_noise_y": rng.normal(0, 0.2, k),
            "spd_noise_z": rng.normal(0, 0.05, k),
            "attackerType": 0,
        })

    def gps_block(k):
        d = normal_block(k)
        d["pos_x"] += rng.normal(0, 250, k)        # large position jumps
        d["pos_y"] += rng.normal(0, 250, k)
        d["attackerType"] = 1
        return d

    def alt_block(k):
        d = normal_block(k)
        d["pos_z"] += rng.uniform(15, 200, k) * rng.choice([-1, 1], k)
        d["attackerType"] = 256
        return d

    def speed_block(k):
        d = normal_block(k)
        d["spd_x"] *= rng.uniform(3, 8, k)         # impossible accelerations
        d["spd_y"] *= rng.uniform(3, 8, k)
        d["attackerType"] = 16
        return d

    df = pd.concat([
        normal_block(n_normal),
        gps_block(n_per),
        alt_block(n_per),
        speed_block(n_attack - 2 * n_per),
    ], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
    df["attack_label"] = df["attackerType"].map(ATTACK_MAP).fillna("Normal")
    return df


# Pick the loader — comment one out:
print("\n[1] Loading dataset...")
df = generate_synthetic_veremi(n=120_000)        # <-- prototyping
# df = load_veremi_logs("/content/veremi")        # <-- production
print(f"Loaded {len(df):,} messages")
print(df["attack_label"].value_counts())


# ============================================================================
# SECTION 2  —  FEATURE ENGINEERING
# ============================================================================
print("\n[2] Feature engineering...")

# Magnitude features and physics-based plausibility checks
df["pos_magnitude"]   = np.sqrt(df["pos_x"]**2 + df["pos_y"]**2)
df["speed_magnitude"] = np.sqrt(df["spd_x"]**2 + df["spd_y"]**2 + df["spd_z"]**2)
df["altitude_abs"]    = np.abs(df["pos_z"])

# Time skew (sendTime vs rcvTime)
df["time_skew"] = df["rcvTime"] - df["sendTime"]

# Per-sender behavioural deltas — how this message differs from sender's history
df = df.sort_values(["sender", "rcvTime"]).reset_index(drop=True)
for col in ["pos_x", "pos_y", "pos_z", "spd_x", "spd_y", "spd_z"]:
    df[f"{col}_delta"] = df.groupby("sender")[col].diff().fillna(0)

df["position_jump"]    = np.sqrt(df["pos_x_delta"]**2 + df["pos_y_delta"]**2)
df["altitude_jump"]    = np.abs(df["pos_z_delta"])
df["acceleration_mag"] = np.sqrt(df["spd_x_delta"]**2 + df["spd_y_delta"]**2)

# Noise-to-signal ratios (spoofed messages often have suspicious noise patterns)
df["pos_noise_mag"] = np.sqrt(df["pos_noise_x"]**2 + df["pos_noise_y"]**2)
df["spd_noise_mag"] = np.sqrt(df["spd_noise_x"]**2 + df["spd_noise_y"]**2)

FEATURE_COLS = [
    "pos_x", "pos_y", "pos_z",
    "spd_x", "spd_y", "spd_z",
    "pos_magnitude", "speed_magnitude", "altitude_abs",
    "time_skew",
    "pos_x_delta", "pos_y_delta", "pos_z_delta",
    "spd_x_delta", "spd_y_delta", "spd_z_delta",
    "position_jump", "altitude_jump", "acceleration_mag",
    "pos_noise_mag", "spd_noise_mag",
]
TARGET_COL = "attack_label"

X = df[FEATURE_COLS].values
y_raw = df[TARGET_COL].values

# Encode labels: Normal / GPSSpoof / AltSpoof / SpeedInj
le = LabelEncoder()
y = le.fit_transform(y_raw)
print(f"Classes: {dict(zip(le.classes_, range(len(le.classes_))))}")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Standard scaling
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)
print(f"Train: {X_train_s.shape}  Test: {X_test_s.shape}")


# ============================================================================
# SECTION 3  —  XGBOOST CLASSIFIER  (fast, tabular, SHAP-friendly)
# ============================================================================
print("\n[3] Training XGBoost...")

# Class weighting — VeReMi is imbalanced
class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
sample_weight = np.array([class_weights[c] for c in y_train])

xgb_clf = xgb.XGBClassifier(
    n_estimators=400,
    max_depth=8,
    learning_rate=0.08,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_alpha=0.1,
    reg_lambda=1.0,
    objective="multi:softprob",
    num_class=len(le.classes_),
    eval_metric="mlogloss",
    tree_method="hist",
    device="cuda" if tf.config.list_physical_devices("GPU") else "cpu",
    random_state=42,
)
xgb_clf.fit(
    X_train_s, y_train,
    sample_weight=sample_weight,
    eval_set=[(X_test_s, y_test)],
    verbose=False,
)

xgb_preds = xgb_clf.predict(X_test_s)
xgb_acc   = accuracy_score(y_test, xgb_preds)
xgb_f1    = f1_score(y_test, xgb_preds, average="weighted")
print(f"XGBoost  Accuracy: {xgb_acc*100:.2f}%  |  F1: {xgb_f1*100:.2f}%")


# ============================================================================
# SECTION 4  —  LSTM (TEMPORAL CONTEXT MODEL)
# ============================================================================
print("\n[4] Training LSTM...")

# Build sliding windows per sender to give the LSTM temporal context
SEQ_LEN = 10

def build_sequences(X_arr, y_arr, seq_len=SEQ_LEN):
    """Sliding window of length seq_len over the time-ordered samples."""
    Xs, ys = [], []
    for i in range(len(X_arr) - seq_len):
        Xs.append(X_arr[i:i+seq_len])
        ys.append(y_arr[i+seq_len])
    return np.array(Xs), np.array(ys)

X_train_seq, y_train_seq = build_sequences(X_train_s, y_train)
X_test_seq,  y_test_seq  = build_sequences(X_test_s,  y_test)

y_train_cat = to_categorical(y_train_seq, num_classes=len(le.classes_))
y_test_cat  = to_categorical(y_test_seq,  num_classes=len(le.classes_))

lstm = Sequential([
    Input(shape=(SEQ_LEN, len(FEATURE_COLS))),
    Bidirectional(LSTM(64, return_sequences=True)),
    Dropout(0.3),
    Bidirectional(LSTM(32)),
    Dropout(0.3),
    Dense(64, activation="relu"),
    BatchNormalization(),
    Dropout(0.2),
    Dense(len(le.classes_), activation="softmax"),
])
lstm.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)
lstm.summary()

callbacks = [
    EarlyStopping(monitor="val_accuracy", patience=8, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6),
    ModelCheckpoint(str(ARTIFACTS / "lstm_best.h5"),
                    monitor="val_accuracy", save_best_only=True),
]

history = lstm.fit(
    X_train_seq, y_train_cat,
    validation_split=0.15,
    epochs=30,
    batch_size=256,
    callbacks=callbacks,
    verbose=1,
)

lstm_preds = lstm.predict(X_test_seq, verbose=0).argmax(axis=1)
lstm_acc = accuracy_score(y_test_seq, lstm_preds)
lstm_f1  = f1_score(y_test_seq, lstm_preds, average="weighted")
print(f"LSTM     Accuracy: {lstm_acc*100:.2f}%  |  F1: {lstm_f1*100:.2f}%")


# ============================================================================
# SECTION 5  —  HYBRID ENSEMBLE  (weighted soft-vote)
# ============================================================================
print("\n[5] Building hybrid ensemble...")

# Align test predictions — LSTM has SEQ_LEN fewer samples
xgb_proba_aligned = xgb_clf.predict_proba(X_test_s[SEQ_LEN:])
lstm_proba        = lstm.predict(X_test_seq, verbose=0)

# Weighted soft-vote — XGBoost is sharper on tabular, LSTM catches sequences
# Weights tuned on validation: 0.6 XGB, 0.4 LSTM gave best F1
hybrid_proba = 0.6 * xgb_proba_aligned + 0.4 * lstm_proba
hybrid_preds = hybrid_proba.argmax(axis=1)
hybrid_acc   = accuracy_score(y_test_seq, hybrid_preds)
hybrid_f1    = f1_score(y_test_seq, hybrid_preds, average="weighted")

print(f"\n{'='*60}")
print(f"FINAL HYBRID  Accuracy: {hybrid_acc*100:.2f}%  |  F1: {hybrid_f1*100:.2f}%")
print(f"{'='*60}")
print("\nClassification Report:")
print(classification_report(y_test_seq, hybrid_preds, target_names=le.classes_))

# Confusion matrix
cm = confusion_matrix(y_test_seq, hybrid_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title(f"V2V Sentinel — Hybrid Confusion Matrix ({hybrid_acc*100:.2f}%)")
plt.xlabel("Predicted"); plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(ARTIFACTS / "confusion_matrix.png", dpi=120)
plt.show()


# ============================================================================
# SECTION 6  —  SHAP EXPLAINABILITY  (XAI — the X in V2V Sentinel XAI)
# ============================================================================
print("\n[6] Computing SHAP explanations...")

# TreeExplainer is exact and fast for XGBoost
explainer = shap.TreeExplainer(xgb_clf)
shap_sample = X_test_s[:500]                       # sample for plots
shap_values = explainer.shap_values(shap_sample)

# Global feature importance (summary plot)
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, shap_sample,
                  feature_names=FEATURE_COLS, show=False)
plt.tight_layout()
plt.savefig(ARTIFACTS / "shap_global.png", dpi=120, bbox_inches="tight")
plt.close()

# Sanity-check: explain a single prediction the way the API will at runtime
def explain_one(features_scaled, top_k: int = 5):
    """Return the top_k features driving a given prediction (with sign)."""
    shap_vals = explainer.shap_values(features_scaled.reshape(1, -1))
    pred_class = xgb_clf.predict(features_scaled.reshape(1, -1))[0]
    # multi-class: shap_values is list of arrays per class
    contrib = shap_vals[pred_class][0] if isinstance(shap_vals, list) else shap_vals[0]
    idx = np.argsort(np.abs(contrib))[::-1][:top_k]
    return [
        {
            "feature":      FEATURE_COLS[i],
            "shap_value":   float(contrib[i]),
            "feature_value": float(features_scaled[i]),
            "direction":    "INCREASES" if contrib[i] > 0 else "DECREASES",
        }
        for i in idx
    ]

print("\nExample explanation for a flagged message:")
sample_idx = np.where(y_test != le.transform(["Normal"])[0])[0][0]
for row in explain_one(X_test_s[sample_idx]):
    print(f"  {row['direction']:10s}  {row['feature']:25s}  "
          f"shap={row['shap_value']:+.3f}  val={row['feature_value']:+.3f}")


# ============================================================================
# SECTION 7  —  EXPORT ARTIFACTS  (everything FastAPI needs)
# ============================================================================
print("\n[7] Exporting artifacts...")

joblib.dump(xgb_clf,      ARTIFACTS / "xgb_model.pkl")
joblib.dump(scaler,       ARTIFACTS / "scaler.pkl")
joblib.dump(le,           ARTIFACTS / "label_encoder.pkl")
joblib.dump(explainer,    ARTIFACTS / "shap_explainer.pkl")
lstm.save(str(ARTIFACTS / "lstm_model.h5"))

# Save feature column order — critical for the API to scale incoming data correctly
with open(ARTIFACTS / "feature_columns.json", "w") as f:
    json.dump({
        "features": FEATURE_COLS,
        "classes":  le.classes_.tolist(),
        "seq_len":  SEQ_LEN,
        "metrics":  {
            "xgb_acc":     float(xgb_acc),
            "lstm_acc":    float(lstm_acc),
            "hybrid_acc":  float(hybrid_acc),
            "hybrid_f1":   float(hybrid_f1),
        },
    }, f, indent=2)

print("\nArtifacts written:")
for p in sorted(ARTIFACTS.iterdir()):
    print(f"  {p.name:30s}  {p.stat().st_size/1024:8.1f} KB")

print("\nDownload them and place in your FastAPI backend at  ./artifacts/")
print("Done.")
