import os
import sys
import time
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import scipy.sparse as sp

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from sklearn.linear_model import PassiveAggressiveClassifier, LogisticRegression, SGDClassifier
from sklearn.calibration import CalibratedClassifierCV

from src.config import config
from src.data.loader import prepare_split_data, load_raw_dataset, get_stratified_splits
from src.models.baselines import build_word_vectorizer, build_char_vectorizer, FakeNewsPipeline
from src.models.lstm_attention import (
    TextVocabulary,
    NewsTorchDataset,
    BiLSTMAttentionClassifier,
    DeepLearningNewsPipeline
)
from src.models.ensemble import StackingEnsembleModel
from src.evaluation.evaluator import evaluate_predictions, print_metrics_summary

def run_training_pipeline():
    start_time = time.time()
    torch.set_num_threads(8)
    print("===================================================================", flush=True)
    print("[*] STARTING DEEP LEARNING & NEURAL ATTENTION TRAINING PIPELINE", flush=True)
    print("    Target Ground Truth: 0 = Fake News, 1 = Real News", flush=True)
    print("===================================================================", flush=True)

    os.makedirs(config.ARTIFACTS_DIR, exist_ok=True)
    
    # 1. Load and prepare stratified data splits
    print("\n[Step 1/6] Loading and preprocessing dataset...", flush=True)
    splits = prepare_split_data(title_repeat=2)
    X_train, y_train = splits['train']
    X_val, y_val = splits['val']
    X_test, y_test = splits['test']

    df_raw = load_raw_dataset()
    raw_splits = get_stratified_splits(df_raw)
    titles_train = raw_splits['train']['title'].fillna('').tolist()
    titles_val   = raw_splits['val']['title'].fillna('').tolist()
    titles_test  = raw_splits['test']['title'].fillna('').tolist()

    print(f"Dataset Partitions:", flush=True)
    print(f"  Train: {len(X_train)} samples", flush=True)
    print(f"  Val:   {len(X_val)} samples", flush=True)
    print(f"  Test:  {len(X_test)} samples (Unseen Holdout)", flush=True)

    # 2. Build Vocabulary for Deep Learning Sequence Model
    print("\n[Step 2/6] Building Text Vocabulary for Neural Sequence Modeling (max 35k words)...", flush=True)
    vocab = TextVocabulary(max_vocab_size=35000)
    vocab.build_vocab(X_train)
    vocab_path = os.path.join(config.ARTIFACTS_DIR, "vocab.json")
    vocab.save(vocab_path)
    print(f"Vocabulary compiled with {len(vocab.word2idx)} tokens and saved to {vocab_path}", flush=True)

    # 3. Train Primary Deep Learning Model: BiLSTM with Bahdanau Attention
    print("\n[Step 3/6] Training Primary Deep Learning Architecture: BiLSTM + Bahdanau Attention...", flush=True)
    max_seq_len = 200
    batch_size = 64
    epochs = 3
    device = "cpu"

    train_ds = NewsTorchDataset(X_train, y_train, vocab, max_len=max_seq_len)
    val_ds = NewsTorchDataset(X_val, y_val, vocab, max_len=max_seq_len)
    test_ds = NewsTorchDataset(X_test, y_test, vocab, max_len=max_seq_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)

    dl_model = BiLSTMAttentionClassifier(
        vocab_size=len(vocab.word2idx),
        embedding_dim=128,
        hidden_dim=128,
        num_layers=1,
        dropout=0.3
    )

    optimizer = torch.optim.AdamW(dl_model.parameters(), lr=2e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.BCEWithLogitsLoss()

    dl_model_path = os.path.join(config.ARTIFACTS_DIR, "bilstm_attention_best.pt")
    best_val_acc = 0.0

    for epoch in range(epochs):
        dl_model.train()
        total_loss = 0.0
        total_samples = 0
        ep_start = time.time()

        for batch in train_loader:
            optimizer.zero_grad()
            logits = dl_model(batch['input_ids'])
            loss = criterion(logits, batch['label'])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(dl_model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item() * len(batch['label'])
            total_samples += len(batch['label'])

        scheduler.step()

        # Validation pass
        dl_model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch in val_loader:
                logits = dl_model(batch['input_ids'])
                preds = (torch.sigmoid(logits) >= 0.5).long()
                val_correct += (preds == batch['label'].long()).sum().item()
                val_total += len(batch['label'])

        val_acc = val_correct / val_total
        print(f"  Epoch {epoch+1}/{epochs} | Train Loss: {total_loss/total_samples:.4f} | Val Acc: {val_acc*100:.2f}% | Time: {time.time()-ep_start:.1f}s", flush=True)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(dl_model.state_dict(), dl_model_path)
            print(f"  --> Saved new best Deep Learning checkpoint to {dl_model_path}", flush=True)

    # Load best checkpoint for holdout testing
    dl_model.load_state_dict(torch.load(dl_model_path, map_location=device))
    dl_model.eval()

    dl_test_preds = []
    dl_test_proba = []
    with torch.no_grad():
        for batch in test_loader:
            logits = dl_model(batch['input_ids'])
            probs = torch.sigmoid(logits).cpu().numpy()
            dl_test_proba.extend(probs)
            dl_test_preds.extend((probs >= 0.5).astype(int))

    dl_test_proba = np.array(dl_test_proba)
    dl_test_preds = np.array(dl_test_preds)

    dl_metrics = evaluate_predictions(
        y_test, 
        dl_test_preds, 
        dl_test_proba, 
        model_name="Deep Learning BiLSTM with Bahdanau Attention"
    )
    print("\n--- Primary Deep Learning Model Evaluation (Holdout Test Set) ---", flush=True)
    print_metrics_summary(dl_metrics)

    # Initialize Deep Learning Pipeline
    dl_pipeline = DeepLearningNewsPipeline(dl_model, vocab, max_len=max_seq_len, device=device)

    # 4. Fit TF-IDF Vectorizers for Baselines & Meta-Ensemble
    print("\n[Step 4/6] Fitting Feature Representations for Baselines & Meta-Ensemble...", flush=True)
    word_vec = build_word_vectorizer(max_features=45000, ngram_range=(1, 2))
    char_vec = build_char_vectorizer(max_features=10000)

    X_train_word = word_vec.fit_transform(X_train)
    X_train_char = char_vec.fit_transform(titles_train)
    X_train_vec = sp.hstack([X_train_word, X_train_char], format='csr')

    X_val_word = word_vec.transform(X_val)
    X_val_char = char_vec.transform(titles_val)
    X_val_vec = sp.hstack([X_val_word, X_val_char], format='csr')

    X_test_word = word_vec.transform(X_test)
    X_test_char = char_vec.transform(titles_test)
    X_test_vec = sp.hstack([X_test_word, X_test_char], format='csr')

    # Train Baseline Model 1: Calibrated Passive-Aggressive
    print("Training Baseline: Calibrated Passive-Aggressive...", flush=True)
    base_pa = PassiveAggressiveClassifier(C=0.5, max_iter=1500, tol=1e-4, random_state=config.RANDOM_SEED)
    clf_pa = CalibratedClassifierCV(estimator=base_pa, method='sigmoid', cv=3)
    clf_pa.fit(X_train_vec, y_train)

    pa_pipeline = FakeNewsPipeline(word_vec, clf_pa, char_vectorizer=char_vec)
    pa_pipeline.save(os.path.join(config.ARTIFACTS_DIR, "model_passive_aggressive.joblib"))

    pa_test_preds = clf_pa.predict(X_test_vec)
    pa_test_proba = clf_pa.predict_proba(X_test_vec)[:, 1]
    pa_metrics = evaluate_predictions(y_test, pa_test_preds, pa_test_proba, model_name="Calibrated Passive-Aggressive (Baseline)")

    # Train Baseline Model 2: Logistic Regression
    print("Training Baseline: Logistic Regression...", flush=True)
    clf_lr = LogisticRegression(C=2.5, max_iter=1500, solver='lbfgs', tol=1e-4, random_state=config.RANDOM_SEED)
    clf_lr.fit(X_train_vec, y_train)

    lr_pipeline = FakeNewsPipeline(word_vec, clf_lr, char_vectorizer=char_vec)
    lr_pipeline.save(os.path.join(config.ARTIFACTS_DIR, "model_logistic_regression.joblib"))

    lr_test_preds = clf_lr.predict(X_test_vec)
    lr_test_proba = clf_lr.predict_proba(X_test_vec)[:, 1]
    lr_metrics = evaluate_predictions(y_test, lr_test_preds, lr_test_proba, model_name="Logistic Regression (Baseline)")

    # 5. Stacking Meta-Ensemble combining Deep Learning + Linear Baselines
    print("\n[Step 5/6] Building Stacking Meta-Ensemble (Deep Learning + Baselines)...", flush=True)
    # Compute DL validation predictions
    dl_val_proba = []
    with torch.no_grad():
        for batch in val_loader:
            logits = dl_model(batch['input_ids'])
            probs = torch.sigmoid(logits).cpu().numpy()
            dl_val_proba.extend(probs)
    dl_val_proba = np.array(dl_val_proba)

    pa_val_proba = clf_pa.predict_proba(X_val_vec)[:, 1]
    lr_val_proba = clf_lr.predict_proba(X_val_vec)[:, 1]

    val_stack = np.column_stack([dl_val_proba, pa_val_proba, lr_val_proba])
    test_stack = np.column_stack([dl_test_proba, pa_test_proba, lr_test_proba])

    ensemble = StackingEnsembleModel(weights=[0.40, 0.35, 0.25])
    ensemble.fit_meta_learner(val_stack, y_val)
    ensemble.save(os.path.join(config.ARTIFACTS_DIR, "stacking_ensemble.joblib"))

    ensemble_test_proba = ensemble.predict_meta(test_stack)
    ensemble_test_preds = (ensemble_test_proba >= 0.5).astype(int)
    ensemble_metrics = evaluate_predictions(
        y_test, 
        ensemble_test_preds, 
        ensemble_test_proba, 
        model_name="Stacking Deep Ensemble (BiLSTM-Att + PA + LR)"
    )
    print_metrics_summary(ensemble_metrics)

    # Save best fallback ML pipeline for backwards compatibility
    pa_pipeline.save(os.path.join(config.ARTIFACTS_DIR, "best_model.joblib"))

    # Save comprehensive benchmark report with Deep Learning as Primary
    all_reports = {
        "primary_production_model": "bilstm_attention_deep_learning",
        "deep_learning_bilstm_attention": dl_metrics,
        "stacking_deep_ensemble": ensemble_metrics,
        "calibrated_passive_aggressive": pa_metrics,
        "logistic_regression": lr_metrics,
        "peak_accuracy": max(dl_metrics['accuracy'], ensemble_metrics['accuracy'], pa_metrics['accuracy']),
        "peak_macro_f1": max(dl_metrics['macro_f1'], ensemble_metrics['macro_f1'], pa_metrics['macro_f1']),
        "model_architecture": "Bidirectional LSTM + Bahdanau Additive Attention with Neural Head",
        "framework": "PyTorch 2.x"
    }

    report_path = os.path.join(config.ARTIFACTS_DIR, "benchmark_metrics.json")
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(all_reports, f, indent=2)
    print(f"\nAll benchmark metrics successfully saved to {report_path}", flush=True)

    # 6. Validation on Generalization and Adversarial Test Cases
    print("\n[Step 6/6] Verifying Deep Learning Model on Generalization Samples...", flush=True)
    demo_samples = [
        ("Federal Reserve (Real)", "Federal Reserve Holds Benchmark Interest Rates Steady Amid Cooling Inflation Data", "The Federal Reserve announced on Wednesday that benchmark interest rates will remain unchanged following a two-day policy meeting, according to official statements and central bank releases."),
        ("Aliens on Hot Wheels (Fake)", "breaking report : the world is invaded my aliens on hot wheels", "the hot wheels brand has collaborated with aliens to inavade the world and start another holocaust"),
        ("James Webb Telescope (Real)", "James Webb Space Telescope Detects Water Vapor in Rocky Planet Formation Zone", "Astronomers using NASAs James Webb Space Telescope have identified clear spectroscopic signatures of water vapor within the inner disk of a young stellar system."),
        ("Miracle Cure Scam (Fake)", "MIRACLE CURE EXPOSED: Big Pharma Panic As Secret Ancient Root Cures All Disease Overnight [MUST SEE]", "Doctors are STUNNED and corrupt pharmaceutical executives are in a panic! This 100% natural ancient herbal remedy is being suppressed because it completely reverses aging and cures every chronic condition in just 24 hours.")
    ]

    for name, title, text in demo_samples:
        fused = f"{title} . {text}"
        res = dl_pipeline.explain_text(fused, top_k=4)
        verdict = "Real News" if res["p_real"] >= 0.5 else "Fake News"
        conf = res["p_real"] if verdict == "Real News" else res["p_fake"]
        print(f"  [{verdict}] {name} -> P(Fake): {res['p_fake']:.4f}, P(Real): {res['p_real']:.4f} (Conf: {conf*100:.1f}%)", flush=True)
        if verdict == "Fake News":
            print(f"      Top Attention Fake Triggers: {[k['token'] for k in res['fake_indicators'][:3]]}", flush=True)
        else:
            print(f"      Top Attention Factual Anchors: {[k['token'] for k in res['real_indicators'][:3]]}", flush=True)

    print("\n=======================================================", flush=True)
    print("[*] DEEP LEARNING MODEL MIGRATION & TRAINING COMPLETE", flush=True)
    print(f" Primary DL Checkpoint: {dl_model_path}")
    print(f" Vocabulary Path:       {vocab_path}")
    print(f" DL Test Accuracy:      {dl_metrics['accuracy'] * 100:.2f}%")
    print(f" Peak Ensemble Accuracy:{all_reports['peak_accuracy'] * 100:.2f}%")
    print(f" Total Runtime:         {time.time() - start_time:.2f} seconds")
    print("=======================================================\n", flush=True)

if __name__ == '__main__':
    run_training_pipeline()
