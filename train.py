import os
import sys
import time
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from sklearn.linear_model import PassiveAggressiveClassifier, LogisticRegression, SGDClassifier
from sklearn.calibration import CalibratedClassifierCV

from src.config import config
from src.data.loader import prepare_split_data
from src.models.baselines import build_vectorizer, FakeNewsPipeline
from src.models.lstm_attention import TextVocabulary, NewsTorchDataset, BiLSTMAttentionClassifier
from src.models.ensemble import StackingEnsembleModel
from src.evaluation.evaluator import evaluate_predictions, print_metrics_summary

def run_training_pipeline():
    start_time = time.time()
    print("===================================================================", flush=True)
    print("[*] STARTING MAXIMUM ACCURACY FAKE NEWS BINARY CLASSIFICATION PIPELINE", flush=True)
    print("===================================================================", flush=True)

    os.makedirs(config.ARTIFACTS_DIR, exist_ok=True)
    
    # 1. Load and prepare stratified data splits
    print("\n[Step 1/6] Loading and preprocessing dataset...", flush=True)
    splits = prepare_split_data(title_repeat=2)
    X_train, y_train = splits['train']
    X_val, y_val = splits['val']
    X_test, y_test = splits['test']
    
    print(f"Dataset Partitions:", flush=True)
    print(f"  Train: {len(X_train)} samples", flush=True)
    print(f"  Val:   {len(X_val)} samples", flush=True)
    print(f"  Test:  {len(X_test)} samples (Unseen Holdout)", flush=True)

    # 2. Vectorize with high-accuracy sublinear TF-IDF
    print("\n[Step 2/6] Fitting high-dimensional sublinear TF-IDF (1-2 N-grams)...", flush=True)
    vectorizer = build_vectorizer(max_features=50000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)
    X_test_vec = vectorizer.transform(X_test)
    print(f"TF-IDF matrix shape: {X_train_vec.shape}", flush=True)

    # 3. Model 1: Calibrated Passive-Aggressive
    print("\n[Step 3/6] Training Model 1: Calibrated Passive-Aggressive...", flush=True)
    base_pa = PassiveAggressiveClassifier(C=0.5, max_iter=1000, random_state=config.RANDOM_SEED)
    clf_pa = CalibratedClassifierCV(estimator=base_pa, method='sigmoid', cv=3)
    clf_pa.fit(X_train_vec, y_train)
    
    pa_pipeline = FakeNewsPipeline(vectorizer, clf_pa)
    pa_pipeline.save(os.path.join(config.ARTIFACTS_DIR, "model_passive_aggressive.joblib"))
    
    pa_test_preds = clf_pa.predict(X_test_vec)
    pa_test_proba = clf_pa.predict_proba(X_test_vec)[:, 1]
    pa_metrics = evaluate_predictions(y_test, pa_test_preds, pa_test_proba, model_name="Calibrated Passive-Aggressive (TF-IDF)")
    print_metrics_summary(pa_metrics)

    # 4. Model 2: Logistic Regression
    print("\n[Step 4/6] Training Model 2: Logistic Regression (L-BFGS)...", flush=True)
    clf_lr = LogisticRegression(C=3.0, max_iter=500, solver='lbfgs', random_state=config.RANDOM_SEED)
    clf_lr.fit(X_train_vec, y_train)
    
    lr_pipeline = FakeNewsPipeline(vectorizer, clf_lr)
    lr_pipeline.save(os.path.join(config.ARTIFACTS_DIR, "model_logistic_regression.joblib"))
    
    lr_test_preds = clf_lr.predict(X_test_vec)
    lr_test_proba = clf_lr.predict_proba(X_test_vec)[:, 1]
    lr_metrics = evaluate_predictions(y_test, lr_test_preds, lr_test_proba, model_name="Logistic Regression (TF-IDF)")
    print_metrics_summary(lr_metrics)

    # 5. Model 3: SGD Log-Loss (Fast Logit)
    print("\n[Step 5/6] Training Model 3: SGD Log-Loss Classifier...", flush=True)
    clf_sgd = SGDClassifier(loss='log_loss', penalty='l2', alpha=1e-5, max_iter=1000, random_state=config.RANDOM_SEED)
    clf_sgd.fit(X_train_vec, y_train)
    
    sgd_pipeline = FakeNewsPipeline(vectorizer, clf_sgd)
    sgd_pipeline.save(os.path.join(config.ARTIFACTS_DIR, "model_sgd_log.joblib"))
    
    sgd_test_preds = clf_sgd.predict(X_test_vec)
    sgd_test_proba = clf_sgd.predict_proba(X_test_vec)[:, 1]
    sgd_metrics = evaluate_predictions(y_test, sgd_test_preds, sgd_test_proba, model_name="SGD Log-Loss (TF-IDF)")
    print_metrics_summary(sgd_metrics)

    # 6. Model 4: Stacking & Soft-Voting Meta-Ensemble
    print("\n[Step 6/6] Building Stacking & Soft-Voting Ensemble...", flush=True)
    pa_val_proba = clf_pa.predict_proba(X_val_vec)[:, 1]
    lr_val_proba = clf_lr.predict_proba(X_val_vec)[:, 1]
    sgd_val_proba = clf_sgd.predict_proba(X_val_vec)[:, 1]
    
    val_stack = np.column_stack([pa_val_proba, lr_val_proba, sgd_val_proba])
    test_stack = np.column_stack([pa_test_proba, lr_test_proba, sgd_test_proba])
    
    ensemble = StackingEnsembleModel(weights=[0.45, 0.45, 0.10])
    ensemble.fit_meta_learner(val_stack, y_val)
    ensemble.save(os.path.join(config.ARTIFACTS_DIR, "stacking_ensemble.joblib"))
    
    ensemble_test_proba = ensemble.predict_meta(test_stack)
    ensemble_test_preds = (ensemble_test_proba >= 0.5).astype(int)
    ensemble_metrics = evaluate_predictions(y_test, ensemble_test_preds, ensemble_test_proba, model_name="Stacking Meta-Ensemble (PA + LR + SGD)")
    print_metrics_summary(ensemble_metrics)

    # Determine Best Model and save as primary production model
    all_models = {
        "Calibrated Passive-Aggressive": (pa_metrics, pa_pipeline),
        "Logistic Regression": (lr_metrics, lr_pipeline),
        "SGD Log-Loss": (sgd_metrics, sgd_pipeline)
    }
    
    best_name = max(all_models, key=lambda k: all_models[k][0]['accuracy'])
    best_metrics, best_pipe = all_models[best_name]
    best_pipe.save(os.path.join(config.ARTIFACTS_DIR, "best_model.joblib"))

    # Save summary report
    all_reports = {
        "calibrated_passive_aggressive": pa_metrics,
        "logistic_regression": lr_metrics,
        "sgd_log_loss": sgd_metrics,
        "stacking_meta_ensemble": ensemble_metrics,
        "best_model_name": best_name,
        "peak_accuracy": best_metrics['accuracy'],
        "peak_macro_f1": best_metrics['macro_f1']
    }
    
    report_path = os.path.join(config.ARTIFACTS_DIR, "benchmark_metrics.json")
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(all_reports, f, indent=2)
    print(f"All benchmark metrics saved to {report_path}", flush=True)

    print("\n=======================================================", flush=True)
    print("[*] FINAL CLASSIFICATION BENCHMARK COMPLETE", flush=True)
    print(f" Best Model:     {best_name}", flush=True)
    print(f" Peak Accuracy:  {best_metrics['accuracy'] * 100:.2f}%")
    print(f" Peak Macro F1:  {best_metrics['macro_f1']:.4f}")
    print(f" Peak ROC-AUC:   {best_metrics['roc_auc']:.4f}")
    print(f" Total Runtime:  {time.time() - start_time:.2f} seconds")
    print("=======================================================\n", flush=True)

if __name__ == '__main__':
    run_training_pipeline()
