# Dataset Study 04: Modeling & Deep Learning Architecture Specification

**Project**: AI-Powered Fake News Detection Using NLP & Deep Learning  
**Dataset**: WELFake (`WELFake_Dataset.csv`)  
**Purpose**: Comprehensive architecture blueprints for ML baselines, Deep Learning networks, Transformer models, and ensemble stacking.

---

## 1. Architecture Suite Overview

```
                      ┌────────────────────────────────────────┐
                      │          WELFake Input Stream          │
                      └──────────────────┬─────────────────────┘
                                         │
     ┌───────────────────┬───────────────┴───────────────┬───────────────────┐
     ▼                   ▼                               ▼                   ▼
┌──────────────┐  ┌──────────────┐                ┌──────────────┐    ┌──────────────┐
│  Tier 1: ML  │  │ Tier 2: DL   │                │ Tier 3: DL   │    │ Tier 4: SOTA │
│  Baselines   │  │ BiLSTM + Att │                │ CNN-BiLSTM   │    │ Transformers │
│ (TF-IDF +    │  │ (GloVe 300d) │                │ (Multi-Head) │    │(RoBERTa/BERT)│
│  LR / PAC)   │  │              │                │              │    │              │
└──────┬───────┘  └──────┬───────┘                └──────┬───────┘    └──────┬───────┘
       │                 │                               │                   │
       └─────────────────┴───────────────┬───────────────┴───────────────────┘
                                         ▼
                      ┌────────────────────────────────────────┐
                      │    Tier 5: Ensemble Stacking Model     │
                      │  (Soft-Voting / Logistic Meta-Learner) │
                      └──────────────────┬─────────────────────┘
                                         ▼
                      ┌────────────────────────────────────────┐
                      │  Final Prediction & Confidence Score   │
                      │   [Fake Probability (0.0 to 1.0)]      │
                      └────────────────────────────────────────┘
```

---

## 2. Model 1: Strong Classical Baseline (TF-IDF + Calibrated Classifier)
- **Vectorization**: TF-IDF with `ngram_range=(1, 3)`, `max_features=40000`, `sublinear_tf=True`.
- **Classifiers**:
  - `PassiveAggressiveClassifier(C=0.5, max_iter=1000)` (Fast, high-dimensional linear boundary).
  - `LogisticRegression(C=2.0, solver='saga', penalty='l2')` with CalibratedClassifierCV.
- **Expected Benchmark**: `~94.5% Accuracy`, `~0.945 Macro-F1`, inference speed `< 1ms/sample`.

---

## 3. Model 2: BiLSTM with Bahdanau Additive Attention
- **Embedding Layer**: Pretrained `GloVe.6B.300d` (frozen for 3 epochs, then fine-tuned).
- **Recurrent Core**: Bidirectional LSTM (2 layers, hidden dimension `128` per direction -> 256 total).
- **Attention Mechanism**:
  $$lpha_t = rac{\exp(v^	op 	anh(W h_t + b))}{\sum_k \exp(v^	op 	anh(W h_k + b))}, \quad c = \sum_t lpha_t h_t$$
- **Dense Head**: `Linear(256 -> 64) -> LayerNorm -> ReLU -> Dropout(0.4) -> Linear(64 -> 1) -> Sigmoid`.
- **Expected Benchmark**: `~96.2% Accuracy`, `~0.962 Macro-F1`.

---

## 4. Model 3: Hybrid Multi-Scale 1D-CNN + BiLSTM Network
- **Intuition**: 1D-CNN captures local n-gram clickbait patterns (filter sizes 3, 4, 5); BiLSTM captures long-range document narrative flow.
- **Architecture**:
  1. Input shape: `(batch_size, 300, 300)`
  2. Parallel Conv1D branches: Kernel sizes `[3, 4, 5]`, filters = `64` each.
  3. MaxPool1D + Concatenation -> Dense feature projection.
  4. Single layer BiLSTM (`hidden_dim=128`, `dropout=0.3`).
  5. Global Average Pooling + Global Max Pooling concatenation.
  6. Final Classification Head with Sigmoid output.

---

## 5. Model 4: Transformer Fine-Tuning (`RoBERTa-base` / `DeBERTa-v3-base`)
- **Base Model**: `roberta-base` (125M parameters) or `microsoft/deberta-v3-base` (86M parameters).
- **Sequence Length**: `512` tokens (`truncation="only_second"`).
- **Optimization**:
  - **Optimizer**: `AdamW(lr=2e-5, eps=1e-8, weight_decay=0.01)`.
  - **Learning Rate Scheduler**: Linear warmup for first 10% steps, followed by Cosine Annealing decay.
  - **Batch Size**: 16 per device (with gradient accumulation steps = 2 -> effective batch size 32).
  - **Mixed Precision**: FP16 / BF16 AMP.
  - **Regularization**: Dropout `0.2`, Label Smoothing `0.05`.
- **Expected Benchmark**: `~98.4% Accuracy`, `~0.984 Macro-F1`, `ROC-AUC > 0.995`.

---

## 6. Model 5: Stacking Meta-Learner
- **Input**: Out-of-fold predicted probability vectors from Baseline, BiLSTM-Att, CNN-BiLSTM, and RoBERTa.
- **Meta-Classifier**: Ridge Logistic Regression / LightGBM.
- **Benefit**: Smooths individual model failure modes, reduces out-of-distribution variance.
