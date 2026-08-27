# 🤖 Master LLM Continuation & Hand-off Guide

> **Purpose for Next AI / Developer:** This document contains the complete state, architecture, dataset mappings, commands, and roadmap for the **VeritasAI: AI Powered Fake News Detection** project. If session tokens exhaust or you switch LLMs, read this file to resume immediately with zero context loss.

---

## 📌 1. Project Overview & Quick Reference

* **Repository:** `AI-Powered-Fake-News-Detection`
* **Stack:** FastAPI (Backend), Scikit-Learn / PyTorch (ML), Vanilla CSS/HTML/JS (Frontend).
* **Deployments:**
  * **Frontend:** Hosted on **Vercel** (`index.html`, `app.js`, `style.css`).
  * **Backend:** Hosted on **Render** (FastAPI running `uvicorn src.serving.app:app`).
* **Ground Truth Convention:**
  * `0 = Fake News / Disinformation / Clickbait`
  * `1 = Real News / Verified / Journalistic Reporting`
* **Artifacts Directory:** `artifacts/` (`best_model.joblib`, `model_logistic_regression.joblib`, `model_sgd_log.joblib`).

---

## 📊 2. Dataset Architecture & Multi-Domain Ingestion

To eliminate domain overfitting, the system uses a multi-domain dataset aggregated into:
📁 `dataset_study/unified_multidomain_dataset.csv` (75,970 balanced samples)

### Data Sources Included:
1. **LIAR Dataset (10,164 samples):** Short political claims & PolitiFact statements (solves the short headline/claim problem).
2. **CoAID Dataset (2,128 samples):** Medical, health, COVID-19, and scientific claims (solves the health/science gap).
3. **WELFake Dataset (63,678 samples):** General long-form journalism and balanced news corpus.

### How to Re-download / Rebuild Datasets:
```bash
python src/data/download_datasets.py
```

---

## 🧠 3. Model & Feature Pipeline Architecture

### Pipeline Components:
1. **Regularized TF-IDF (Sublinear TF, `min_df=5`, `max_df=0.90`, 35k features):** Prevents single-token shortcut memorization.
2. **Journalistic & Scientific Attribution Extractor (`src/data/preprocessor.py`):**
   * Detects peer-reviewed citations (`published in the journal`, `researchers found`, `astronomers using`).
   * Detects institutional attribution (`unanimous vote`, `official statement`, `spokesperson said`).
   * Detects sensationalist clickbait triggers (`world is on fire`, `miracle cure`, `secret plot`, `MUST SEE`, all-caps).
3. **Hybrid Credibility Fusion (`src/serving/api.py`):** Combines model probability with domain-invariant attribution scores to ensure high accuracy on unseen out-of-distribution news.
4. **Live Knowledge-Grounded Fact-Checking (`src/llm_reasoner/fact_check_agent.py`):** Real-time encyclopedic grounding (Wikipedia Open API) to corroborate factual claims against real-world entities.

---

## ⚡ 4. Standard Operational Commands

### 1. Retrain the Regularized Multi-Domain Model:
```bash
python retrain_robust_model.py
```

### 2. Run Generalization & Out-of-Distribution Benchmark Tests:
```bash
python test_sample.py
```

### 3. Verify Dataset Integrity & Test Metrics:
```bash
python verify_dataset_model.py
```

### 4. Start Local Development Server:
```bash
uvicorn src.serving.app:app --reload --port 8000
```

### 5. Push Updates to Production (Vercel & Render):
```bash
git add .
git commit -m "your message"
git push origin main
```

---

## 🚀 5. Immediate Next Steps / Roadmap for Successor AI

If the user asks to continue improving the project further, here is the exact recommended roadmap:

1. **Sentence-Transformers Fine-Tuning (`all-MiniLM-L6-v2`):**
   * Script a PyTorch fine-tuning loop using HuggingFace `sentence-transformers` on `dataset_study/unified_multidomain_dataset.csv`.
   * Save ONNX weights for fast CPU inference under 40ms.
2. **Google Fact Check Tools API Key Integration:**
   * Allow user to optionally provide `GOOGLE_FACTCHECK_API_KEY` in `.env` to query certified fact-checkers (Snopes, PolitiFact, Reuters Fact Check).
3. **User Feedback Loop Endpoint (`/feedback`):**
   * Add a lightweight SQLite database or file logger in FastAPI to record user-reported false positives for active learning retrain cycles.
