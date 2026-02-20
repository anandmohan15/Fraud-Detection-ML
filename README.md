# 🔍 Fraud Detection Using Machine Learning

A complete end-to-end machine learning pipeline that generates synthetic transaction data, engineers features, trains a Random Forest classifier, evaluates performance, and visualizes feature importance — all in a single Python script.

---

## 📁 Project Structure

```
LTI2/
├── fraud_detection.py      # Main script — runs the full pipeline
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── transactions.csv        # Generated synthetic dataset (after running)
├── confusion_matrix.png    # Confusion matrix heatmap (after running)
└── feature_importance.png  # Feature importance chart (after running)
```

---

## 🚀 How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the full pipeline
python fraud_detection.py
```

The script will:
1. Generate `transactions.csv` (10,000 transactions)
2. Engineer features and encode categoricals
3. Train a Random Forest model
4. Print evaluation metrics to the console
5. Save `confusion_matrix.png` and `feature_importance.png`

---

## 🎯 Approach Overview

### Problem Statement
Credit card / digital payment fraud costs billions annually. We build a supervised ML model to classify transactions as **Normal** or **Fraud** using synthetically generated data that mimics real-world fraud patterns.

### Pipeline Steps

| Step | Description |
|------|-------------|
| **Data Generation** | 10,000 synthetic transactions with ~3 % fraud rate |
| **Feature Engineering** | Time features, user aggregates, amount deviation, one-hot encoding |
| **Model Training** | Random Forest with balanced class weights (80/20 split) |
| **Evaluation** | Precision, Recall, F1-score, ROC-AUC, Confusion Matrix |
| **Explainability** | Feature importance bar chart |

---

## 🕵️ Fraud Patterns Used

We inject three realistic fraud patterns into the synthetic data:

### 1. High Transaction Amount Spikes
Normal transactions range from **$10 – $500**, while fraudulent transactions range from **$500 – $5,000**. This simulates real-world scenarios where stolen cards are used for large purchases.

### 2. Rapid Transactions in Short Time
Fraudsters make **3–5 transactions within 5 minutes** in a burst. This mimics the real pattern where criminals quickly drain accounts before the fraud is detected.

### 3. Shared Device Across Multiple Users
A small pool of **10 "fraud devices"** is shared across different user IDs. In reality, fraudsters often reuse the same phone or computer to commit fraud under different identities.

---

## 🤖 Model Choice Rationale — Why Random Forest?

| Reason | Explanation |
|--------|-------------|
| **Handles imbalanced data** | With `class_weight='balanced'`, it adjusts for the 97/3 class split |
| **No feature scaling needed** | Tree-based models work directly with raw feature values |
| **Captures non-linear patterns** | Fraud patterns (amount spikes, rapid bursts) are non-linear |
| **Built-in feature importance** | Makes the model explainable out of the box |
| **Robust to overfitting** | Ensemble of 200 trees with controlled depth (max_depth=15) |
| **Beginner-friendly** | Simple API, easy to understand and tune |

---

## 📊 Evaluation Results

After running the pipeline, you will see results similar to the following (exact values may vary slightly):

| Metric | Score |
|--------|-------|
| **Precision** | ~0.85+ |
| **Recall** | ~0.80+ |
| **F1-Score** | ~0.82+ |
| **ROC-AUC** | ~0.95+ |

### Confusion Matrix
The confusion matrix heatmap (`confusion_matrix.png`) shows:
- **True Positives** — correctly detected frauds
- **True Negatives** — correctly identified normal transactions
- **False Positives** — normal transactions flagged as fraud
- **False Negatives** — missed frauds (most critical)

### Feature Importance
The feature importance chart (`feature_importance.png`) reveals which features contribute most to fraud detection. Typically **amount**, **amount_deviation**, and **user_avg_amount** rank highest.

---

## 🔑 Key Findings

1. **Transaction amount is the strongest predictor** — fraud transactions have significantly higher amounts than normal ones.
2. **User-level aggregates add value** — features like `amount_deviation` (how much a transaction deviates from a user's average) help the model distinguish between unusual and normal behaviour.
3. **Time features contribute moderately** — hour-of-day and day-of-week provide some signal, as fraud patterns may cluster at certain times.
4. **Class balancing is critical** — without `class_weight='balanced'`, the model would be biased toward predicting "Normal" for every transaction due to the 97/3 imbalance.

---

## ⚠️ Limitations

1. **Synthetic data** — the fraud patterns are hand-crafted and may not capture the full complexity of real-world fraud.
2. **No temporal modelling** — we do not use sequence-based models (e.g., LSTM) that could capture the order of transactions over time.
3. **Limited feature set** — real systems use hundreds of features including IP addresses, browser fingerprints, merchant risk scores, etc.
4. **Static model** — no online learning or model retraining as new data arrives.
5. **No cost-sensitive learning** — in production, the cost of missing a fraud (False Negative) is far higher than a false alarm (False Positive).

---

## 🚀 Future Improvements

1. **Use real-world datasets** (e.g., Kaggle's IEEE-CIS Fraud Detection dataset) for more realistic modelling.
2. **Try advanced models** like XGBoost, LightGBM, or neural networks for potentially better performance.
3. **Add SMOTE** (Synthetic Minority Over-sampling Technique) as an alternative to class weighting.
4. **Implement sequence-based features** — time gaps between consecutive transactions per user.
5. **Build a real-time scoring API** using Flask or FastAPI for production deployment.
6. **Add cross-validation** for more robust performance estimates.
7. **Use SHAP values** for instance-level explainability (why was *this specific transaction* flagged?).

---

## 🛠️ Dependencies

- Python 3.8+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

---

## 📜 License

This project is created for educational / academic purposes.
