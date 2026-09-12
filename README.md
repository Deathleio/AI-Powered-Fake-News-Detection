# 🧠 VeritasAI: Deep Learning Fake News & Veracity Intelligence Platform

An enterprise-grade, explainable **Deep Learning (DL)** and **Natural Language Processing (NLP)** platform for detecting fake news, media disinformation, unverified breaking claims, and clickbait sensationalism.

Built on **PyTorch**, the primary classification core features a **Bidirectional LSTM (BiLSTM) with Bahdanau Additive Attention (`BiLSTMAttentionClassifier`)**, providing high-accuracy sequence modeling paired with token-level neural attention explainability, real-time press wire corroboration, and domain credibility scoring.

---

## 📋 Table of Contents

- [Deep Learning Architecture Overview](#-deep-learning-architecture-overview)
- [Key Features](#-key-features)
- [Model Benchmark & Performance](#-model-benchmark--performance)
- [Explainable AI: Bahdanau Attention Mechanism](#-explainable-ai-bahdanau-attention-mechanism)
- [System Requirements](#-system-requirements)
- [Quick Start Guide](#-quick-start-guide)
  - [1. Clone & Setup](#1-open-terminal--navigate-to-project)
  - [2. Virtual Environment](#2-create-and-activate-virtual-environment)
  - [3. Install Dependencies](#3-install-dependencies)
- [How to Run the Program Locally](#-how-to-run-the-program-locally)
  - [Option A: Instant Terminal CLI Analysis](#option-a-instant-terminal-cli-analysis)
  - [Option B: Run the Web Server & Full Dashboard](#option-b-run-the-web-server--full-dashboard)
  - [Option C: REST API Endpoints & Grounding Queries](#option-c-rest-api-endpoints--grounding-queries)
- [API Reference & Grounding Endpoints](#-api-reference--grounding-endpoints)
- [Running Automated Tests](#-running-automated-tests)
- [Model Training & Retraining](#-model-training--retraining)
- [Technical Documentation & Architecture](#-technical-documentation--architecture)
- [Project Structure](#-project-structure)

---

## 🧠 Deep Learning Architecture Overview

Unlike shallow bag-of-words or simple count-based keyword heuristics, VeritasAI processes articles as **temporal semantic sequences** through a deep neural pipeline:

```
                          ┌────────────────────────────────────────┐
                          │   News Input (Headline + Body Text)    │
                          └──────────────────┬─────────────────────┘
                                             │
                                  [Text Preprocessor]
                                             │
                          ┌──────────────────▼─────────────────────┐
                          │       Word Vocabulary Encoder          │
                          │        (35,000 Learned Tokens)         │
                          └──────────────────┬─────────────────────┘
                                             │
                          ┌──────────────────▼─────────────────────┐
                          │      Token Embedding Layer             │
                          │   (128-dimensional dense vectors)      │
                          └──────────────────┬─────────────────────┘
                                             │
                          ┌──────────────────▼─────────────────────┐
                          │    Bidirectional LSTM Core             │
                          │ (Forward & Backward Temporal Context)  │
                          └──────────────────┬─────────────────────┘
                                             │
                          ┌──────────────────▼─────────────────────┐
                          │    Bahdanau Additive Attention         │
                          │   α_t = Softmax(v^T * tanh(W * h_t))   │
                          └──────────────────┬─────────────────────┘
                                             │
                          ┌──────────────────▼─────────────────────┐
                          │    Deep Dense Classification Head      │
                          │  LayerNorm ➔ ReLU ➔ Dropout ➔ Linear   │
                          └──────────────────┬─────────────────────┘
                                             │
               ┌─────────────────────────────┴─────────────────────────────┐
               ▼                                                           ▼
┌─────────────────────────────┐                             ┌─────────────────────────────┐
│    Binary Veracity Score    │                             │  Attention-Saliency Map     │
│ P(Real News) vs P(Fake News)│                             │ Token-Level Attribution XAI │
└─────────────────────────────┘                             └─────────────────────────────┘
```

### 1. Primary Model: `BiLSTMAttentionClassifier`
- **Embedding Layer**: 128-dimensional learned dense embeddings with spatial 2D dropout.
- **Recurrent Core**: Bidirectional LSTM capturing forward narrative trajectory ($\vec{h}_t$) and reverse syntactic structure ($\overleftarrow{h}_t$).
- **Bahdanau Additive Attention**: Computes normalized attention weights $\alpha_t$ across all word tokens, pooling temporal hidden states into an information-dense context vector:
  $$\alpha_t = \frac{\exp(v^\top \tanh(W h_t))}{\sum_{k=1}^T \exp(v^\top \tanh(W h_k))}, \quad c = \sum_{t=1}^T \alpha_t h_t$$
- **Classification Head**: Multi-layer neural projection with Layer Normalization, ReLU activation, and Dropout regularization.

### 2. Multi-Scale CNN-BiLSTM Hybrid (`CNNBiLSTMClassifier`)
- Multi-scale parallel 1D convolutions (kernel sizes 3, 4, 5) capturing local n-gram clickbait phrases concatenated into a BiLSTM for global narrative coherence.

### 3. Stacking Deep Ensemble (`StackingEnsembleModel`)
- Fuses out-of-fold probability distributions from the primary Deep Learning sequence model with calibrated linear and stylistic learners, achieving peak classification accuracy and variance reduction.

---

## ✨ Key Features

- **Deep Learning Sequence Intelligence**: Understands context, syntax, and narrative flow rather than relying merely on word counts.
- **Intrinsic Token-Level Explainability (XAI)**: Directly outputs learned Bahdanau Attention scores for every word, visually highlighting deceptive clickbait triggers vs. credibility-anchoring tokens.
- **Dedicated Live Press Wire Grounding (`/api/v1/ground-claim`)**: Cross-references claims against live Google News Wire RSS feeds, major international agencies (Reuters, AP, Bloomberg, BBC), and Wikipedia factual knowledge.
- **Mixed Veracity & Breakthrough Detection**: Flags uncorroborated "breakthrough" assertions (e.g. secret cures, alien discoveries) that lack wire corroboration despite real-world background topics.
- **Domain Reputation Registry**: Evaluates 100+ global journalistic domains, satire publications (e.g. *The Onion*), and state-sponsored disinformation outlets.
- **Enterprise REST API**: Built on FastAPI with complete OpenAPI / Swagger documentation, JSON schema validation, and health monitoring.

---

## 📊 Model Benchmark & Performance

Evaluated on **11,396 unseen holdout articles** from the balanced multi-domain dataset (`artifacts/benchmark_metrics.json`):

| Model Architecture | Framework | Accuracy | Macro F1 | ROC-AUC | Primary Role |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **BiLSTM with Bahdanau Attention** | **PyTorch 2.x** | **90.00%** | **0.8994** | **0.9687** | **Active Production Core** |
| **Stacking Deep Ensemble (BiLSTM + Baselines)** | **PyTorch / Scikit** | **93.57%** | **0.9353** | **0.9872** | **Peak Ensemble Pipeline** |
| Calibrated Passive-Aggressive | Scikit-Learn | 92.48% | 0.9246 | 0.9780 | Fast Lightweight Fallback |
| Regularized Logistic Regression | Scikit-Learn | 91.90% | 0.9188 | 0.9734 | Comparative Baseline |

*Ground Truth Convention: `0 = Fake News / Clickbait`, `1 = Real / Factual News`.*

---

## 🔍 Explainable AI: Bahdanau Attention Mechanism

Unlike black-box deep learning classifiers, VeritasAI provides **white-box neural interpretability**:

```python
# The model produces logits AND attention weights:
logits, att_weights = model(input_tensor, return_attention=True)
# att_weights represents the exact softmax probability mass the neural network
# allocated to each individual word in the sentence.
```

- **Fake News Detection**: Top attended words highlight alarmist rhetoric, conspiracy formulation, or urgent medical claims (`"secret ancient root"`, `"big pharma panic"`, `"globalist plot"`).
- **Real News Detection**: Top attended words highlight authoritative attributions, institutional sources, and dates (`"announced"`, `"unanimous vote"`, `"central bank"`, `"published in the journal"`).

---

## 💻 System Requirements

- **Python**: 3.9, 3.10, 3.11, 3.12, or 3.13
- **Deep Learning Framework**: PyTorch `>= 2.0.0`
- **Operating System**: Windows 10/11, macOS, or Linux
- **Hardware**: Runs efficiently on standard CPUs (<15ms per inference); optionally supports CUDA GPUs.

---

## 🚀 Quick Start Guide

### 1. Open Terminal & Navigate to Project

```bash
cd "c:/AI Powered Fake News Detection"
```

### 2. Create and Activate Virtual Environment

#### On Windows (PowerShell):
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

#### On Windows (CMD):
```cmd
python -m venv venv
venv\Scripts\activate.bat
```

#### On macOS / Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🖥️ How to Run the Program Locally

### Option A: Instant Terminal CLI Analysis

Analyze news articles or run the benchmark demo directly in your terminal:

#### 1. Run the Built-in Demo Benchmark (4 Real-World News Cases)
```bash
python test_sample.py --demo
```
*Outputs confidence scores, Bahdanau Attention keywords, and multi-engine fact-checking rationales for verified news, sensational clickbait, space exploration, and medical scams.*

#### 2. Analyze Custom Headline & Text
```bash
python test_sample.py --title "NASA James Webb Space Telescope Detects Water Vapor in Rocky Planet Formation Zone" --text "Astronomers identified clear spectroscopic signatures of water within the inner disk of a young star system."
```

---

### Option B: Run the Web Server & Full Dashboard

Launch the unified FastAPI backend and local web dashboard on a single port:

```bash
python run_pipeline.py --mode serve --host 127.0.0.1 --port 8000
```
*(Or directly: `python -m src.serving.app`)*

- 🌐 **Interactive Web Dashboard**: **[http://127.0.0.1:8000](http://127.0.0.1:8000)**
- 📖 **Interactive OpenAPI / Swagger Docs**: **[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)**

---

### Option C: REST API Endpoints & Grounding Queries

Query the active Deep Learning server via `curl` or PowerShell:

#### 1. Inspect Deep Learning Model Info (`/api/v1/model-info`)
```bash
curl http://127.0.0.1:8000/api/v1/model-info
```
**Response:**
```json
{
  "engine_name": "VeritasAI Deep Learning Veracity Platform",
  "primary_model_type": "Deep Learning (PyTorch)",
  "architecture": "Bidirectional LSTM with Bahdanau Additive Attention",
  "framework": "PyTorch 2.x",
  "explainability_engine": "Bahdanau Attention Token Saliency",
  "parameters": {
    "vocab_size": 35000,
    "sequence_length": 256,
    "device": "cpu",
    "attention_mechanism": "Additive Bahdanau [v^T * tanh(W * h)]"
  },
  "benchmark_summary": {
    "dl_test_accuracy": 0.9000,
    "dl_macro_f1": 0.8994,
    "dl_roc_auc": 0.9687
  }
}
```

#### 2. Verify and Ground a Specific Claim (`/api/v1/ground-claim`)
```bash
curl -X POST "http://127.0.0.1:8000/api/v1/ground-claim" \
  -H "Content-Type: application/json" \
  -d "{\"claim\": \"Federal Reserve holds benchmark interest rates steady\"}"
```
**Response:**
```json
{
  "claim": "Federal Reserve holds benchmark interest rates steady",
  "cleaned_search_query": "Federal Reserve holds benchmark interest rates steady",
  "corroboration_score": 0.954,
  "has_wire_corroboration": true,
  "has_claim_corroboration": true,
  "grounding_verdict": "Corroborated by Verified Press Wires",
  "confidence_percentage": 98.5
}
```

#### 3. Deep Learning Prediction (`/predict`)
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d "{\"title\": \"Secret cure suppressed by doctors\", \"text\": \"Miracle remedy heals all conditions in 24 hours!\"}"
```

#### 4. Attention Saliency & Full XAI (`/explain`)
```bash
curl -X POST "http://127.0.0.1:8000/explain" \
  -H "Content-Type: application/json" \
  -d "{\"title\": \"James Webb Telescope detects water vapor\", \"text\": \"Spectroscopic analysis confirms water vapor in planet forming disk.\"}"
```

---

## 📡 API Reference & Grounding Endpoints

| Endpoint | Method | Purpose | Deep Learning / Engine Feature |
| :--- | :---: | :--- | :--- |
| `/health` | `GET` | System health check | Verifies API runtime and service status |
| `/api/v1/model-info` | `GET` | Model transparency | Reports active DL architecture, PyTorch version, parameters & attention heads |
| `/api/v1/ground-claim` | `POST` | Claim grounding | Cross-checks assertions against live Google News Wire RSS & Wikipedia |
| `/predict` | `POST` | Quick classification | BiLSTM sequence prediction with calibrated hybrid probability |
| `/explain` | `POST` | Full XAI diagnostic | Attention saliency weights, claim breakdown, live news corroboration & trust score |
| `/api/v1/analyze-url` | `POST` | Scrape & evaluate URL | Extracts headline, body, and domain reputation for instant verification |
| `/api/v1/feedback` | `POST` | Active learning | Logs analyst corrections for active learning retraining |
| `/api/v1/export-report`| `POST` | Forensic report | Generates cryptographically hashed audit certificates |

---

## 🧪 Running Automated Tests

Run the automated test suite covering unit operations, PyTorch models, forward passes, attention extraction, and API endpoints:

```bash
python -m unittest discover tests
```

Expected output:
```
[VeritasAI] Primary Deep Learning BiLSTM-Attention pipeline successfully loaded into memory.
Ran 12 tests in 1.6s
OK
```

---

## 🔄 Model Training & Retraining

To train the complete Deep Learning pipeline on the full dataset:

```bash
python train.py
```

This will:
1. Load and stratify the 75,970-article dataset.
2. Compile a 35,000-word neural vocabulary into `artifacts/vocab.json`.
3. Train `BiLSTMAttentionClassifier` on CPU/GPU with AdamW, CosineAnnealingLR, and gradient clipping.
4. Save best checkpoint to `artifacts/bilstm_attention_best.pt`.
5. Train comparative baselines and Stacking Meta-Ensemble.
6. Evaluate holdout test metrics and record to `artifacts/benchmark_metrics.json`.

---

## 📚 Technical Documentation & Architecture

Comprehensive documentation is provided in the `docs/` folder:

- 🏛️ **[System Architecture (docs/ARCHITECTURE.md)](docs/ARCHITECTURE.md)**: Deep dive into the BiLSTM recurrent core, Bahdanau Attention equations, multi-scale CNN-BiLSTM, live news grounding engine, and domain credibility registry.
- 📋 **[Product Requirements Document (docs/PRD.md)](docs/PRD.md)**: Product specifications, SLAs, 3-tier classification spectrum, and verification criteria.
- 📜 **[Development & Model Evolution (development timeline.md)](development%20timeline.md)**: Chronological model journey from early linear models to Generation 4 ("The Explainable Neural Attention Architecture").
- 🔬 **[Dataset Studies (dataset_study/)](dataset_study/)**: Statistical profiling of WELFake, LIAR, and CoAID datasets.

---

## 📂 Project Structure

```
AI Powered Fake News Detection/
├── artifacts/
│   ├── bilstm_attention_best.pt  # Primary trained PyTorch Deep Learning model
│   ├── vocab.json                # Learned 35,000-token sequence vocabulary
│   ├── benchmark_metrics.json    # Holdout test accuracy and benchmark report
│   ├── best_model.joblib         # Secondary/fallback baseline model
│   └── stacking_ensemble.joblib  # Meta-ensemble stacking model
├── dataset_study/                # Balanced multi-domain dataset studies
├── docs/                         # Architecture blueprints and PRD
├── frontend/                     # Modern responsive HTML5/CSS/JS frontend
├── src/
│   ├── config.py                 # Hyperparameters, sequence lengths & paths
│   ├── credibility/              # Domain reputation registry
│   ├── data/                     # Text preprocessing, wire sanitization, URL scraper
│   ├── evaluation/               # Metrics evaluator (Accuracy, F1, ROC-AUC)
│   ├── explainability/           # Bahdanau Attention saliency & claim segmentation
│   ├── llm_reasoner/             # Live press wire corroboration & Wikipedia grounding
│   ├── models/                   # Deep Learning models (BiLSTM-Att, CNN-BiLSTM)
│   └── serving/                  # FastAPI REST API & dashboard launcher
├── tests/                        # Automated unit and integration tests
├── requirements.txt              # Production dependencies (including PyTorch)
├── train.py                      # Production Deep Learning training pipeline
├── test_sample.py                # Terminal CLI evaluation script
└── README.md                     # Primary documentation
```

---

## 📄 License & Academic Citation

Designed for academic research, journalistic fact-checking, and enterprise disinformation mitigation. Built with **PyTorch**, **FastAPI**, and modern NLP sequence modeling.
