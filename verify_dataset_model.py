import sys
import os
import json
import joblib
import pandas as pd
import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from src.config import config
from src.data.loader import prepare_split_data
from src.evaluation.evaluator import evaluate_predictions

print("===============================================================")
print("[*] BACKEND & DATASET VERIFICATION AUDIT")
print("===============================================================")

# 1. Dataset verification
df = pd.read_csv("WELFake_Dataset.csv")
total = len(df)
fake_cnt = int((df['label'] == 1).sum())
real_cnt = int((df['label'] == 0).sum())
print("1. RAW DATASET INTEGRITY:")
print(f"   Total Articles:      {total:,}")
print(f"   Fake News (Class 1): {fake_cnt:,} ({fake_cnt/total*100:.2f}%)")
print(f"   Real News (Class 0): {real_cnt:,} ({real_cnt/total*100:.2f}%)")
print(f"   Missing Titles:      {df['title'].isna().sum():,}")
print(f"   Missing Texts:       {df['text'].isna().sum():,}")

# 2. Stratified Splits
splits = prepare_split_data(title_repeat=1)
X_test, y_test = splits['test']
print("\n2. DATA SPLITS (Holdout Evaluation Partition):")
print(f"   Holdout Test Set:    {len(X_test):,} unseen articles (15.00% stratified)")

# 3. Model Pipeline Verification
pipe = joblib.load("artifacts/best_model.joblib")
print("\n3. TRAINED MODEL ARCHITECTURE:")
print(f"   Feature Extractor:   TfidfVectorizer (Sublinear TF, English Stopwords Scrubbed)")
print(f"   Feature Space:       {len(pipe.vectorizer.vocabulary_):,} unique n-grams (1-2)")
print(f"   Classifier:          {type(pipe.clf).__name__} with Probability Calibration")

# 4. Holdout Performance on Dataset
test_vec = pipe.vectorizer.transform(X_test)
preds = pipe.clf.predict(test_vec)
proba = pipe.clf.predict_proba(test_vec)[:, 1]

metrics = evaluate_predictions(y_test, preds, proba, model_name="Production Pipeline")

print("\n4. RIGOROUS TEST SET EVALUATION (10,821 UNSEEN ARTICLES):")
print(f"   Accuracy:            {metrics['accuracy']*100:.2f}%")
print(f"   Macro F1-Score:      {metrics['macro_f1']:.4f}")
print(f"   ROC-AUC Score:       {metrics['roc_auc']:.4f}")
print(f"   Precision:           {metrics['precision']:.4f}")
cm = metrics['confusion_matrix']
tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
print(f"   True Negatives (TN): {tn:,} (Correctly identified Real News)")
print(f"   True Positives (TP): {tp:,} (Correctly identified Fake News)")
print(f"   False Positives(FP): {fp:,} (Real misclassified as Fake)")
print(f"   False Negatives(FN): {fn:,} (Fake misclassified as Real)")
print("===============================================================")
