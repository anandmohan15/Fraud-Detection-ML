"""
=============================================================================
 FRAUD DETECTION USING MACHINE LEARNING
 ----------------------------------------
 A complete end-to-end ML pipeline that:
   1. Generates synthetic transaction data with realistic fraud patterns
   2. Engineers meaningful features from raw data
   3. Trains a Random Forest classifier
   4. Evaluates model performance
   5. Visualizes feature importance for explainability

 Author : LTI2 Project
 Python : 3.8+
=============================================================================
"""

# ── Imports ──────────────────────────────────────────────────────────────────
import os
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)

import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend (works without a display)
import matplotlib.pyplot as plt
import seaborn as sns

# ── Reproducibility ─────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)


# =============================================================================
# PART 1 — SYNTHETIC DATA GENERATION
# =============================================================================
def generate_synthetic_data(num_transactions: int = 10_000,
                            fraud_rate: float = 0.03) -> pd.DataFrame:
    """
    Generate synthetic transaction data with realistic fraud patterns.

    Fraud patterns injected:
      1. High-amount spikes   — fraud transactions have unusually large amounts.
      2. Rapid transactions   — fraudsters make several transactions in minutes.
      3. Shared device usage  — the same device is used by multiple users.

    Parameters
    ----------
    num_transactions : int
        Total number of transactions to generate (default 10,000).
    fraud_rate : float
        Approximate fraction of fraudulent transactions (default 0.03 = 3 %).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: transaction_id, user_id, merchant_id, amount,
        timestamp, location, payment_method, device_id, is_fraud.
    """
    print("=" * 70)
    print("PART 1 — Generating Synthetic Transaction Data")
    print("=" * 70)

    # ── Configuration ────────────────────────────────────────────────────
    num_users     = 500          # pool of unique users
    num_merchants = 100          # pool of unique merchants
    num_devices   = 400          # pool of unique devices
    locations     = ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix",
                     "Philadelphia", "San Antonio", "San Diego", "Dallas", "Austin"]
    payment_methods = ["credit_card", "debit_card", "UPI", "net_banking", "wallet"]

    # How many transactions are fraudulent?
    num_fraud  = int(num_transactions * fraud_rate)
    num_normal = num_transactions - num_fraud

    # ── Helper: generate a random timestamp in the last 90 days ──────────
    base_time = datetime(2025, 1, 1)
    def random_timestamp():
        return base_time + timedelta(seconds=random.randint(0, 90 * 24 * 3600))

    # ── Generate NORMAL transactions ─────────────────────────────────────
    normal_records = []
    for i in range(num_normal):
        normal_records.append({
            "transaction_id": f"TXN_{i:06d}",
            "user_id":        f"USER_{random.randint(1, num_users):04d}",
            "merchant_id":    f"MERCH_{random.randint(1, num_merchants):03d}",
            "amount":         round(random.uniform(10, 500), 2),    # normal range
            "timestamp":      random_timestamp(),
            "location":       random.choice(locations),
            "payment_method": random.choice(payment_methods),
            "device_id":      f"DEV_{random.randint(1, num_devices):04d}",
            "is_fraud":       0,
        })

    # ── Generate FRAUDULENT transactions ─────────────────────────────────
    # We create a small pool of "fraud devices" shared across users (Pattern 3)
    fraud_devices = [f"DEV_{random.randint(1, num_devices):04d}" for _ in range(10)]

    fraud_records = []
    idx = num_normal  # continue transaction IDs

    while len(fraud_records) < num_fraud:
        # Pick a random user and a shared fraud device
        user_id   = f"USER_{random.randint(1, num_users):04d}"
        device_id = random.choice(fraud_devices)           # Pattern 3: shared device

        # Pattern 2: rapid burst — 3 to 5 transactions within minutes
        burst_size  = random.randint(3, 5)
        burst_start = random_timestamp()

        for b in range(burst_size):
            if len(fraud_records) >= num_fraud:
                break
            fraud_records.append({
                "transaction_id": f"TXN_{idx:06d}",
                "user_id":        user_id,
                "merchant_id":    f"MERCH_{random.randint(1, num_merchants):03d}",
                "amount":         round(random.uniform(500, 5000), 2),  # Pattern 1: high amount
                "timestamp":      burst_start + timedelta(seconds=random.randint(0, 300)),  # within 5 min
                "location":       random.choice(locations),
                "payment_method": random.choice(payment_methods),
                "device_id":      device_id,
                "is_fraud":       1,
            })
            idx += 1

    # ── Combine, shuffle, and reset index ────────────────────────────────
    df = pd.DataFrame(normal_records + fraud_records)
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    # Save to CSV
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "transactions.csv")
    df.to_csv(csv_path, index=False)

    # Print summary
    print(f"  Total transactions : {len(df)}")
    print(f"  Fraudulent         : {df['is_fraud'].sum()}  "
          f"({df['is_fraud'].mean() * 100:.2f} %)")
    print(f"  Saved to           : {csv_path}\n")

    return df


# =============================================================================
# PART 2 — FEATURE ENGINEERING
# =============================================================================
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create new features from the raw transaction data.

    New features:
      • hour               — hour of the day extracted from timestamp
      • day_of_week        — day of the week (0 = Monday … 6 = Sunday)
      • user_avg_amount    — average transaction amount per user
      • amount_deviation   — how much this transaction deviates from the user's avg
      • txn_count_per_user — total transaction count per user

    Categorical columns (location, payment_method) are one-hot encoded.

    Parameters
    ----------
    df : pd.DataFrame
        Raw transaction DataFrame.

    Returns
    -------
    pd.DataFrame
        Feature-engineered DataFrame ready for modelling.
    """
    print("=" * 70)
    print("PART 2 — Feature Engineering")
    print("=" * 70)

    # ── Convert timestamp to datetime (if not already) ───────────────────
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # ── Time-based features ──────────────────────────────────────────────
    df["hour"]        = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek

    # ── User-level aggregation features ──────────────────────────────────
    user_stats = df.groupby("user_id")["amount"].agg(
        user_avg_amount="mean",
        txn_count_per_user="count"
    ).reset_index()

    df = df.merge(user_stats, on="user_id", how="left")

    # Amount deviation = how far this transaction is from the user's average
    df["amount_deviation"] = df["amount"] - df["user_avg_amount"]

    print(f"  Created features : hour, day_of_week, user_avg_amount, "
          f"amount_deviation, txn_count_per_user")

    # ── One-hot encode categorical features ──────────────────────────────
    df = pd.get_dummies(df, columns=["location", "payment_method"], drop_first=True)
    print(f"  One-hot encoded  : location, payment_method")

    # ── Drop columns not used for modelling ──────────────────────────────
    drop_cols = ["transaction_id", "user_id", "merchant_id",
                 "timestamp", "device_id"]
    df.drop(columns=drop_cols, inplace=True)

    print(f"  Dropped columns  : {drop_cols}")
    print(f"  Final shape      : {df.shape}\n")

    return df


# =============================================================================
# PART 3 — MODEL DEVELOPMENT
# =============================================================================
def train_model(df: pd.DataFrame):
    """
    Train a Random Forest classifier on the processed data.

    • Uses an 80 / 20 train-test split.
    • Applies class_weight='balanced' to handle the 97 / 3 class imbalance.

    Parameters
    ----------
    df : pd.DataFrame
        Feature-engineered DataFrame (output of engineer_features).

    Returns
    -------
    model : RandomForestClassifier
        Trained model.
    X_test : pd.DataFrame
        Test feature matrix.
    y_test : pd.Series
        True labels for the test set.
    feature_names : list[str]
        Names of the features used for training.
    """
    print("=" * 70)
    print("PART 3 — Model Training (Random Forest)")
    print("=" * 70)

    # ── Separate features and label ──────────────────────────────────────
    X = df.drop(columns=["is_fraud"])
    y = df["is_fraud"]
    feature_names = list(X.columns)

    # ── Train-test split (80 / 20) ──────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    print(f"  Training samples : {len(X_train)}")
    print(f"  Testing samples  : {len(X_test)}")

    # ── Train Random Forest ──────────────────────────────────────────────
    model = RandomForestClassifier(
        n_estimators=200,             # number of trees
        max_depth=15,                 # limit tree depth to avoid overfitting
        class_weight="balanced",      # handle class imbalance automatically
        random_state=SEED,
        n_jobs=-1,                    # use all CPU cores
    )
    model.fit(X_train, y_train)
    print(f"  Model trained    : RandomForestClassifier (200 trees, max_depth=15)")
    print()

    return model, X_test, y_test, feature_names


# =============================================================================
# PART 4 — EVALUATION
# =============================================================================
def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model and display metrics + confusion matrix.

    Metrics reported:
      • Precision, Recall, F1-score (per class and weighted average)
      • ROC-AUC score
      • Confusion matrix heatmap (saved as confusion_matrix.png)

    Parameters
    ----------
    model : RandomForestClassifier
        Trained model.
    X_test : pd.DataFrame
        Test feature matrix.
    y_test : pd.Series
        True labels for the test set.
    """
    print("=" * 70)
    print("PART 4 — Model Evaluation")
    print("=" * 70)

    # ── Predictions ──────────────────────────────────────────────────────
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]   # probability of fraud

    # ── Metrics ──────────────────────────────────────────────────────────
    precision = precision_score(y_test, y_pred)
    recall    = recall_score(y_test, y_pred)
    f1        = f1_score(y_test, y_pred)
    roc_auc   = roc_auc_score(y_test, y_proba)

    print(f"\n  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  F1-score  : {f1:.4f}")
    print(f"  ROC-AUC   : {roc_auc:.4f}\n")

    # ── Full classification report ───────────────────────────────────────
    print("  Classification Report:")
    print("  " + "-" * 55)
    report = classification_report(y_test, y_pred, target_names=["Normal", "Fraud"])
    for line in report.split("\n"):
        print(f"  {line}")
    print()

    # ── Confusion Matrix Heatmap ─────────────────────────────────────────
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Normal", "Fraud"],
                yticklabels=["Normal", "Fraud"])
    plt.title("Confusion Matrix", fontsize=14, fontweight="bold")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    cm_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"  Confusion matrix saved to: {cm_path}\n")

    return precision, recall, f1, roc_auc


# =============================================================================
# PART 5 — EXPLAINABILITY (Feature Importance)
# =============================================================================
def plot_feature_importance(model, feature_names):
    """
    Plot and save a horizontal bar chart of the top 15 feature importances.

    Parameters
    ----------
    model : RandomForestClassifier
        Trained model.
    feature_names : list[str]
        Names of the features used during training.
    """
    print("=" * 70)
    print("PART 5 — Feature Importance (Explainability)")
    print("=" * 70)

    # ── Get importances and sort ─────────────────────────────────────────
    importances = model.feature_importances_
    feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=True)

    # Show top 15 features
    top_n = feat_imp.tail(15)

    # ── Plot ─────────────────────────────────────────────────────────────
    plt.figure(figsize=(8, 6))
    top_n.plot(kind="barh", color=sns.color_palette("viridis", len(top_n)))
    plt.title("Top 15 Feature Importances (Random Forest)", fontsize=14, fontweight="bold")
    plt.xlabel("Importance Score")
    plt.ylabel("Feature")
    plt.tight_layout()
    fi_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "feature_importance.png")
    plt.savefig(fi_path, dpi=150)
    plt.close()

    print(f"  Top 5 features:")
    for feat, score in feat_imp.tail(5).iloc[::-1].items():
        print(f"    • {feat:30s}  {score:.4f}")
    print(f"\n  Feature importance chart saved to: {fi_path}\n")


# =============================================================================
# MAIN — Run the full pipeline
# =============================================================================
def main():
    """Run all five parts of the fraud detection pipeline sequentially."""
    print("\n" + "=" * 70)
    print("   FRAUD DETECTION -- END-TO-END ML PIPELINE")
    print("=" * 70 + "\n")

    # Part 1: Generate data
    df_raw = generate_synthetic_data(num_transactions=10_000, fraud_rate=0.03)

    # Part 2: Feature engineering
    df_processed = engineer_features(df_raw.copy())

    # Part 3: Train model
    model, X_test, y_test, feature_names = train_model(df_processed)

    # Part 4: Evaluate
    precision, recall, f1, roc_auc = evaluate_model(model, X_test, y_test)

    # Part 5: Feature importance
    plot_feature_importance(model, feature_names)

    # ── Final Summary ────────────────────────────────────────────────────
    print("=" * 70)
    print("PIPELINE COMPLETE — Summary")
    print("=" * 70)
    print(f"  Dataset          : transactions.csv  (10,000 rows)")
    print(f"  Model            : Random Forest (200 trees)")
    print(f"  Precision        : {precision:.4f}")
    print(f"  Recall           : {recall:.4f}")
    print(f"  F1-score         : {f1:.4f}")
    print(f"  ROC-AUC          : {roc_auc:.4f}")
    print(f"  Outputs saved    : confusion_matrix.png, feature_importance.png")
    print("=" * 70)


if __name__ == "__main__":
    main()
