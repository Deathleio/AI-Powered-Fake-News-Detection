# Dataset Study 05: Evaluation Metrics & Explainability (XAI) Specification

**Project**: AI-Powered Fake News Detection Using NLP & Deep Learning  
**Dataset**: WELFake (`WELFake_Dataset.csv`)  
**Purpose**: Validation strategy, evaluation metric definitions, model calibration, explainable AI (XAI) protocols, and adversarial robustness testing.

---

## 1. Cross-Validation & Data Partitioning Strategy

To prevent synthetic performance inflation and data leakage:

```
Full WELFake Dataset (72,134 records)
   │
   ▼
[Data Cleaning & Deduplication] ───► Clean Corpus (~71,800 records)
   │
   ├── 70% Training Split (~50,260 samples) ────► Model Training & Gradient Updates
   │
   ├── 15% Validation Split (~10,770 samples) ──► Early Stopping & Hyperparameter Tuning
   │
   └── 15% Holdout Test Split (~10,770 samples) ─► Final Unbiased Evaluation & XAI
```

- **Stratification**: Enforce strict class stratification across all three splits (`label` 0 vs 1 distribution preserved exactly).
- **Random Seed**: Pin `random_state = 42` for total experimental reproducibility.

---

## 2. Comprehensive Evaluation Metrics

| Metric | Mathematical Formula | Target Threshold | Primary Role |
| :--- | :--- | :--- | :--- |
| **Accuracy** | $\frac{TP + TN}{TP + TN + FP + FN}$ | $\ge 96.0\%$ | High-level overall correctness |
| **Precision (Fake / 1)** | $\frac{TP}{TP + FP}$ | $\ge 96.5\%$ | Minimizes False Positives (flagging real news as fake) |
| **Recall (Fake / 1)** | $\frac{TP}{TP + FN}$ | $\ge 96.5\%$ | Minimizes False Negatives (missing malicious fake news) |
| **Macro F1-Score** | $\frac{2 \cdot P_{macro} \cdot R_{macro}}{P_{macro} + R_{macro}}$ | $\ge 96.0\%$ | Core optimization metric |
| **ROC-AUC** | $\int_0^1 \text{TPR}(t) d\text{FPR}(t)$ | $\ge 0.990$ | Discriminative threshold independence |
| **PR-AUC** | Area under Precision-Recall Curve | $\ge 0.985$ | Class-specific precision integrity |
| **ECE (Calibration)** | $\sum_{m=1}^M \frac{|B_m|}{N} |\text{acc}(B_m) - \text{conf}(B_m)|$ | $\le 0.040$ | Reliable confidence score calibration |

---

## 3. Explainability (XAI) Suite

Explainability is essential for users and fact-checkers to trust the model's classifications.

### A. Token Saliency via Integrated Gradients (Deep Learning / Transformers)
- Computes path integrals of model gradients with respect to input word embeddings from a zero baseline:
  $$	ext{IG}_i(x) = (x_i - x_i') 	imes \int_0^1 rac{\partial F(x' + lpha(x - x'))}{\partial x_i} dlpha$$
- Highlights exact words/phrases that pushed prediction towards Fake (Red) or Real (Green).

### B. LIME (Local Interpretable Model-agnostic Explanations)
- Perturbs input text by masking tokens and fits an interpretable sparse linear surrogate model locally around the prediction.
- Fast, visual, and model-agnostic (works for Classical ML, BiLSTM, and LLMs).

### C. Self-Attention Heatmap Visualization (Transformers)
- Extracts the multi-head attention weights from the final layer `[CLS]` token across all input tokens to reveal long-distance contextual dependencies.

---

## 4. Robustness & Adversarial Testing Matrix

Before deployment, models must be evaluated against 4 stress tests:
1. **Headline-Only Evaluation**: Assess model accuracy when `text` is absent.
2. **Body-Only Evaluation**: Assess model accuracy when `title` is absent.
3. **Paraphrase Invariance Test**: Perturb clickbait keywords with neutral synonyms to verify the model is not overfitting to shallow triggers.
4. **Source Anonymization Test**: Verify accuracy remains stable when news organization names are redacted.
