# Architecture — VeritasAI Enterprise Veracity Intelligence Platform

## 1. High-Level App Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                 Client Frontend (Vercel / Netlify / Browser)                │
│     index.html  +  style.css  +  app.js (Vanilla JS + FontAwesome Icons)    │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │ HTTP POST /explain or /api/v1/analyze-url
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│              FastAPI Backend Application (src/serving/api.py)               │
├─────────────────────────────────────────────────────────────────────────────┤
│  1. Ingestion & Extraction Layer:                                           │
│     - Raw text & headline ingestion                                         │
│     - 4-Tier Web URL Extractor (JSON-LD -> OpenGraph -> DOM -> Meta)        │
│                                                                             │
│  2. NLP Preprocessing & Feature Engineering:                                │
│     - Dateline / wire agency watermark sanitizer (strips shortcut leakage)  │
│     - Title-body weighted fusion (`fuse_title_body`)(Concatenation)                         │
│     - Stylistic & psycholinguistic feature extractor (ALL CAPS, triggers)   │
│                                                                             │
│  3. Claim Decomposition & Forensic Analysis:                                │
│     - Sentence-level Claim Segmenter (`claim_segmenter.py`)                 │
│     - Extraordinary Breakthrough Assertion matcher (regex patterns)         │
│     - Mixed-Veracity / Hybrid Disinformation profile evaluator              │
│                                                                             │
│  4. Knowledge Grounding & Corroboration (Concurrent Thread Pool):           │
│     - Real-Time News Wire Corroboration Engine (`news_grounding_engine.py`) │
│     - Live Google News Open Search XML Feed parsing                         │
│     - Open Wikipedia Encyclopedic Grounding with Refutation Cue Detection   │
│     - Global Publisher Credibility Registry (`domain_registry.py`)          │
│                                                                             │
│  5. Statistical & Neural Model Inference:                                   │
│     - Production: Dual-Vectorizer (Word 1-2g + Char 3-4g) + Linear Pipeline │
│     - Deep Learning / Attention: BiLSTM-Attention (`lstm_attention.py`)     │
│                                                                             │
│  6. Hybrid Probability Arbitration (`compute_hybrid_fake_probability`):     │
│     - Merges statistical ML probability + domain authority + live wire      │
│       corroboration + stylistic risk + mixed-veracity claims                │
│                                                                             │
│  7. Explainability & Token Saliency:                                        │
│     - TF-IDF feature contribution extraction (positive=real, negative=fake) │
│     - High-impact token chips & highlighted HTML generation                 │
│     - Deterministic multi-source fact-checking rationale synthesis          │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │ JSON Response (ExplainablePredictionResponse)
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       Forensic Verification Report                          │
│   • 3-Tier Verdict: Real News | Partially Fake / Misleading | Fake News     │
│   • Veritas Trust Score (0 - 100) & Confidence Percentage                   │
│   • Sentence-by-Sentence Claim Breakdown with Risk Tags                     │
│   • Live Wire Corroboration & Encyclopedic Cross-References                 │
│   • Publisher Domain Authority Badge & Bias Rating                          │
│   • Salient Token Triggers & Cryptographically Signed Audit Report          │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Detailed Request Path (`/explain` & `/api/v1/analyze-url`)
1. **Request Intake:** Frontend posts `{ title, text, source_url }` to `/explain`, or `{ url }` to `/api/v1/analyze-url`.
2. **URL Extraction (if URL provided):** `extract_article_from_url()` executes a 4-tier scraper (JSON-LD schema -> OpenGraph tags -> BeautifulSoup DOM `<article>` -> meta description), extracting sanitized title, body text, author, and publication date.
3. **Data Preprocessing:**
   - `sanitize_wire_leakage()` strips wire agency datelines (e.g., `WASHINGTON (Reuters) -`, `LONDON (AP) --`) and brand watermarks to prevent spurious shortcut learning.
   - `fuse_title_body()` concatenates title and body text with weighted title repetition.
4. **Forensic Feature Extraction:**
   - `extract_stylistic_features()` calculates casing ratios, exclamation mark density, sensational clickbait keywords (*"miracle cure"*, *"bombshell"*, *"secret plot"*), and legitimate institutional attribution markers (*"according to"*, *"officials confirmed"*).
   - `segment_and_analyze_claims()` parses the article into atomic sentences, categorizing them (*High-Risk Sensational Claim*, *Unverified Breakthrough Assertion*, *Verified Sourced Statement*, *Empirical Data Point*).
   - `analyze_mixed_veracity_profile()` checks for hybrid disinformation (scientific or technical context blended with unverified claims).
5. **Statistical & Model Scoring:**
   - The production pipeline (`FakeNewsPipeline`) computes raw probability $P(\text{fake})$ and $P(\text{real})$ using dual word and character n-gram feature matrices. Convention: **Index 0 = Fake News, Index 1 = Real News**.
6. **Concurrent Live Grounding:**
   - `fetch_encyclopedic_corroboration()` queries open Wikipedia search for entity-level refutations and encyclopedic context.
   - `fetch_live_news_corroboration()` queries open Google News search feeds to match claim keywords against Tier 1 press wires (Reuters, AP, BBC, Bloomberg).
7. **Hybrid Arbitration (`compute_hybrid_fake_probability`):**
   - Combines statistical ML probability with domain reputation, wire corroboration scores, sensationalism flags, and claim risk ratios.
   - Assigns 3-tier verdict: `Real News`, `Partially Fake / Misleading`, or `Fake News`.
8. **Explainability & Rationale:**
   - `extract_tfidf_word_importance()` extracts directional token contributions.
   - `generate_highlighted_html()` renders inline visual text highlights.
   - `LLMFactCheckReasoner.synthesize_verdict()` compiles the structured narrative explanation.
   - Calculates the **Veritas Trust Score** ($0$ = severe disinformation, $100$ = verified authenticity; mixed veracity capped at $15\text{--}35$).

---

## 2. Complete Folder & File Map

```
c:/AI Powered Fake News Detection/
├── docs/                          # Architecture blueprints & product requirements
│   ├── ARCHITECTURE.md            # System architecture, pipeline flows, model blueprints
│   └── PRD.md                     # Product requirements document (v2.0 Enterprise)
├── README.md                      # Comprehensive developer and user guide
│
├── frontend/                      # Static web client (Netlify / Vercel ready)
│   ├── index.html                 # Modern responsive dashboard, presets, and modals
│   ├── style.css                  # Design tokens, risk badges, responsive grid layout
│   ├── app.js                     # API client, live status check, async DOM renderer
│   ├── vercel.json                # Vercel deployment and routing rules
│   └── _redirects                 # Netlify SPA redirect rules
│
├── src/                           # Python enterprise backend package
│   ├── __init__.py
│   ├── config.py                  # Global configurations, file paths, hyperparameters
│   │
│   ├── credibility/               # Publisher reputation & authority
│   │   ├── __init__.py
│   │   └── domain_registry.py     # Database of 30+ outlets across 5 tiers (Wires, Press, Satire, Disinfo)
│   │
│   ├── data/                      # Dataset ingestion, scraping, and preprocessing
│   │   ├── __init__.py
│   │   ├── loader.py              # CSV loader, label normalizer (0=Fake, 1=Real), stratified splits
│   │   ├── preprocessor.py        # Leakage sanitizer, text fusion, stylistic feature extraction
│   │   ├── url_extractor.py       # 4-tier web article scraper (JSON-LD, OpenGraph, DOM, Meta)
│   │   └── download_datasets.py   # Multi-domain dataset fetcher (WELFake + LIAR + CoAID unification)
│   │
│   ├── evaluation/                # Performance assessment & metrics
│   │   ├── __init__.py
│   │   └── evaluator.py           # Accuracy, Macro/Binary F1, ROC-AUC, confusion matrix
│   │
│   ├── explainability/            # Forensic explainability & claim analysis
│   │   ├── __init__.py
│   │   ├── token_saliency.py      # TF-IDF word importance weights and HTML snippet highlighter
│   │   └── claim_segmenter.py     # Sentence tokenizer, extraordinary claim matcher, mixed-veracity profile
│   │
│   ├── llm_reasoner/              # Knowledge grounding & reasoning synthesis
│   │   ├── __init__.py
│   │   ├── news_grounding_engine.py # Live Google News XML feed cross-corroboration engine
│   │   └── fact_check_agent.py    # Multi-engine synthesizer with Wikipedia refutation cues
│   │
│   ├── models/                    # Modeling architectures (Classical, Neural Attention, Ensembles)
│   │   ├── __init__.py
│   │   ├── baselines.py           # Dual TF-IDF vectorizers (word+char) and FakeNewsPipeline
│   │   ├── lstm_attention.py      # PyTorch BiLSTM with Bahdanau Additive Attention mechanism
│   │   ├── cnn_bilstm.py          # PyTorch Multi-Scale 1D-CNN + BiLSTM hybrid architecture
│   │   └── ensemble.py            # Soft-voting and Stacking meta-ensemble model
│   │
│   └── serving/                   # API serving & development runtime
│       ├── __init__.py
│       ├── api.py                 # FastAPI application, routes, CORS, hybrid arbitration
│       └── app.py                 # Local server runner and fallback web dashboard
│
├── artifacts/                     # Serialized models and benchmark reports (git-ignored)
│   ├── best_model.joblib          # Active production pipeline (Dual TF-IDF + Calibrated Linear)
│   ├── model_logistic_regression.joblib
│   ├── model_passive_aggressive.joblib
│   ├── model_sgd_log.joblib
│   ├── stacking_ensemble.joblib
│   ├── bilstm_attention_best.pt   # Trained PyTorch BiLSTM-Attention checkpoint
│   ├── vocab.json                 # Vocabulary dictionary for deep learning models
│   └── benchmark_metrics.json     # Benchmark evaluation report (98.7% test accuracy)
│
├── dataset_study/                 # Analytical research and architectural specifications
│   ├── 01_dataset_schema_and_integrity.md
│   ├── 02_statistical_and_eda_profile.md
│   ├── 03_nlp_preprocessing_and_tokenization_spec.md  # Tokenization for ML, RNN, Transformers
│   ├── 04_modeling_and_deep_learning_architecture.md  # Architecture blueprints (RoBERTa, BiLSTM, CNN)
│   ├── 05_evaluation_metrics_and_explainability.md
│   ├── 06_multi_agent_system_and_llm_orchestration.md
│   ├── active_learning_feedback.jsonl
│   └── unified_multidomain_dataset.csv (WELFake + LIAR + CoAID: 75,970 articles)
│
├── tests/                         # Pytest test suite
│   ├── __init__.py
│   ├── test_api.py                # Endpoint validation tests
│   ├── test_models.py             # Model prediction and vectorizer shape tests
│   ├── test_preprocessor.py       # Dateline sanitization and stylistic feature tests
│   ├── test_live_grounding.py     # Live Google News XML feed parsing tests
│   ├── test_mixed_claims.py       # Comprehensive test suite for mixed-veracity claims
│   └── test_url_pipeline.py       # 4-tier URL scraping tests
│
├── train.py                       # Benchmark training script (trains baselines & ensemble)
├── retrain_robust_model.py        # Regularized retraining script for anti-overfitting
├── run_pipeline.py                # Command-line helper to train or run server
├── verify_dataset_model.py        # Model and dataset integrity verification
├── cm_audit.py                    # Confusion matrix error analysis
├── test_sample.py                 # Quick inference testing utility
│
├── requirements.txt               # Production Python dependencies
├── Dockerfile                     # Container deployment image recipe
├── Procfile                       # Process launcher for cloud PaaS (Render / Railway)
├── render.yaml                    # Render infrastructure configuration
├── netlify.toml                   # Netlify static hosting configuration
└── WELFake_Dataset.csv            # Primary training dataset (72,134 news articles, ~245 MB)
```

---

## 3. Detailed Modeling Stack: Classical, Neural Attention & Transformers

The project architecture accommodates three distinct modeling paradigms:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          FACTCHECK Modeling Stack                           │
├────────────────────────────────┬────────────────────────────────────────────┤
│ Route A: Production Serving    │ • Dual TF-IDF (Word 1-2g + Char 3-4g)      │
│ (Active in artifacts/)         │ • Calibrated Passive-Aggressive / LogReg   │
│                                │ • Sub-5ms inference, <50MB RAM             │
│                                │ • 98.72% Holdout Test Accuracy             │
├────────────────────────────────┼────────────────────────────────────────────┤
│ Route B: Neural Sequence &     │ • `BiLSTMAttentionClassifier`              │
│ Attention Modeling (PyTorch)   │ • `BahdanauAttention` additive mechanism   │
│                                │ • `CNNBiLSTMClassifier` multi-scale hybrid │
│                                │ • Learns context vectors across time steps │
├────────────────────────────────┼────────────────────────────────────────────┤
│ Route C: Transformer           │ • `RoBERTa-base` / `DeBERTa-v3-base`       │
│ Fine-Tuning Specification      │ • Dual-segment tokenizer (512 token limit) │
│ (dataset_study/ specifications)│ • Multi-Head Self-Attention layers         │
│                                │ • Designed for GPU-enabled deployments     │
└────────────────────────────────┴────────────────────────────────────────────┘
```

### Route A — Production Serving Model (`FakeNewsPipeline`)
- **Dual Vectorizer:** Combines word n-grams ($1\text{--}2$ grams, 50k features) on fused text with character n-grams ($3\text{--}4$ grams, 10k features, case-sensitive) on titles. Captures both topic semantics and stylistic signals (e.g., casing, punctuation density, typos).
- **Classifiers:** Calibrated Passive-Aggressive Classifier (sigmoid calibration via 3-fold CV) and regularized Logistic Regression ($C=0.8$, $L_2$ penalty).
- **Deployment Advantage:** Runs at **sub-5 millisecond inference latency** on standard CPUs without GPU dependencies, consuming under 50MB of RAM. This prevents memory-overflow restarts on free-tier hosting platforms (such as Render's 512MB RAM ceiling).

### Route B — Neural Attention & Deep Learning (`src/models/lstm_attention.py`, `cnn_bilstm.py`)
- **PyTorch BiLSTM with Bahdanau Attention (`BiLSTMAttentionClassifier`):**
  - **Embedding Layer:** 128-dimensional learned embeddings with 2D spatial dropout.
  - **Bidirectional LSTM:** 2-layer recurrent network generating forward and backward hidden states:
    $$\vec{h}_t = \text{LSTM}(\vec{h}_{t-1}, x_t), \quad \overleftarrow{h}_t = \text{LSTM}(\overleftarrow{h}_{t+1}, x_t), \quad h_t = [\vec{h}_t \,;\, \overleftarrow{h}_t]$$
  - **Bahdanau Additive Attention:** Dynamically calculates attention scores and weights over hidden states to generate a focused context vector $c$:
    $$e_t = v^T \tanh(W h_t), \quad \alpha_t = \frac{\exp(e_t)}{\sum_{k=1}^T \exp(e_k)}, \quad c = \sum_{t=1}^T \alpha_t h_t$$
  - **Classification Head:** Multi-layer feed-forward network with LayerNorm, ReLU, and dropout.
- **CNN-BiLSTM Hybrid (`CNNBiLSTMClassifier`):**
  - Parallel 1D temporal convolutions with kernel sizes 3, 4, and 5 to capture multi-scale local phrase patterns.
  - Concatenated convolutional activations fed into a bidirectional LSTM for global sequence modeling, followed by temporal max-pooling and dense classification.

### Route C — Transformer Specification (`RoBERTa-base` / `DeBERTa-v3-base`)
- Detailed in [`dataset_study/03_nlp_preprocessing_and_tokenization_spec.md`](file:///c:/AI%20Powered%20Fake%20News%20Detection/dataset_study/03_nlp_preprocessing_and_tokenization_spec.md) and [`dataset_study/04_modeling_and_deep_learning_architecture.md`](file:///c:/AI%20Powered%20Fake%20News%20Detection/dataset_study/04_modeling_and_deep_learning_architecture.md).
- **Tokenizer:** Dual-segment tokenizer preserving the headline and truncating only the body:
  `tokenizer(title, text, max_length=512, truncation="only_second")`.
- **Architecture:** Pre-trained 12-layer multi-head self-attention transformer backbone with a pooled classification head.
- **Role in the Project:** Represents the high-compute transformer benchmark blueprint for dedicated GPU servers. The lightweight dual-vectorizer linear model was selected for production serving to maintain sub-second response times and zero cloud operational costs.

---

## 4. Multi-Tier Veracity & Arbitration Architecture

The core innovation in VeritasAI v2.0 is the **Hybrid Probability Arbitration Engine** (`compute_hybrid_fake_probability` in [`src/serving/api.py`](file:///c:/AI%20Powered%20Fake%20News%20Detection/src/serving/api.py)):

```
Statistical Model Probability (0.0 to 1.0)
                     │
                     ▼
       ┌─────────────────────────────┐
       │   Stylistic Risk Penalty    │  ALL CAPS (+0.40), Clickbait Triggers (+0.28)
       └─────────────┬───────────────┘
                     │
                     ▼
       ┌─────────────────────────────┐
       │  Publisher Domain Authority │  Tier 1 Wires (-50%), Satire (min 0.85 Fake)
       └─────────────┬───────────────┘
                     │
                     ▼
       ┌─────────────────────────────┐
       │ Live Wire Corroboration     │  Reuters/AP Matches (-80% Fake Probability)
       │ (Google News + Wikipedia)   │  Topic covered but claim absent (min 0.76 Fake)
       └─────────────┬───────────────┘
                     │
                     ▼
       ┌─────────────────────────────┐
       │ Mixed-Veracity Claim Engine │  Extraordinary assertion present -> flags as
       │                             │  "Partially Fake / Misleading" (Veritas score 28)
       └─────────────┬───────────────┘
                     │
                     ▼
     Final Calibrated Veracity Verdict & Veritas Trust Score (0-100)
```

---

## 5. Serving & Deployment Topology

```
User Browser
    │
    ▼
Vercel / Netlify (Static Hosting)
    │  Fetches API (auto-detects localhost:8000 or production Render URL)
    ▼
Render / Railway PaaS (FastAPI / Uvicorn)
    │  Executes pipeline, scrapes URLs, queries Google News/Wikipedia
    ▼
Artifacts Cache (`artifacts/best_model.joblib`)
    │  Loaded lazily once at startup, held in memory for sub-5ms evaluation
```

- **Target Label Convention:** `0 = Fake News / Disinformation`, `1 = Real News / Grounded Journalism`.
- **Stateless & Scalable:** API instances are completely stateless, requiring no persistent database for inference. Optional active learning logs are appended to `active_learning_feedback.jsonl`.