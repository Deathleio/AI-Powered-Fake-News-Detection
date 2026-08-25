# Architecture — AI Fake News Detection

## 1. High-Level App Flow

```
┌────────────┐   HTTP   ┌───────────────────────┐   load   ┌─────────────────────┐
│  Frontend  │ ───────► │   FastAPI Backend     │ ───────► │  Trained Artifacts   │
│ (Vue/Vue)  │  /explain │                       │          │  best_model.joblib   │
│ index.html │ ◄─────── │   src/serving/api.py  │ ◄─────── │  (joblib/pickle)     │
└────────────┘  JSON    │                       │          └─────────────────────┘
                         │   ┌────────────────┐ │
                         │   │ Orchestration  │ │
                         │   │ fusion → vec → │ │
                         │   │ predict → rand │ │
                         │   └────────────────┘ │
                         └──────────────────────┘
```

### Detailed request path (`/explain`)
1. Frontend posts `{ title, text }` to `/explain`.
2. `fuse_title_body()` concatenates title + body (with `title_repeat=1`).
3. `extract_stylistic_features()` computes style signals (all-caps, sensational keywords, attribution score).
4. `model.predict_proba()` yields P(real) and P(fake). Convention: **index 0 = fake, index 1 = real**.
5. `extract_tfidf_word_importance()` returns top fake/real token indicators.
6. `generate_highlighted_html()` renders the annotated snippet.
7. `LLMFactCheckReasoner.synthesize_verdict()` builds the plain-language rationale.
8. Backend returns JSON in `ExplainablePredictionResponse` shape.

## 2. Folder / File Structure

```
c:/AI Powered Fake News Detection/
├── PRD.md                    # product requirements
├── Architecture.md           # this file
├── rules.md                  # AI coding rules & boundaries
├── Phases.md                 # phased delivery roadmap
├── Design.md                 # visual identity & tokens
├── memory.md                 # progress tracker (updated regularly)
│
├── frontend/                 # static Netlify/static-hostable UI
│   ├── index.html            # layout, preset buttons
│   ├── style.css             # design tokens + responsive theme
│   ├── app.js                # API client, retries, rendering
│   ├── vercel.json
│   └── _redirects
│
├── src/                      # Python backend package
│   ├── __init__.py
│   ├── config.py             # Config dataclass (paths, hyps, split ratios)
│   ├── data/
│   │   ├── loader.py         # CSV load, cleaning, stratified splits
│   │   └── preprocessor.py   # fusing, stylistic feature extraction
│   ├── models/
│   │   ├── baselines.py      # vectorizer, FakeNewsPipeline
│   │   ├── cnn_bilstm.py     # deep CNN+BiLSTM architecture
│   │   ├── lstm_attention.py # BiLSTM w/ attention
│   │   └── ensemble.py        # stacking meta-ensemble
│   ├── evaluation/
│   │   └── evaluator.py      # metrics (acc/F1/ROC/etc.)
│   ├── explainability/
│   │   └── token_saliency.py # TF-IDF word importance + HTML highlight
│   ├── llm_reasoner/
│   │   └── fact_check_agent.py # rule-based rationale synthesizer
│   └── serving/
│       ├── api.py            # FastAPI app, routes, CORS, pydantic schemas
│       └── app.py            # local launcher + static mount, fallback UI
│
├── artifacts/                 # trained model pickles/joblib (git-ignored)
│   ├── best_model.joblib
│   ├── model_passive_aggressive.joblib
│   ├── model_logistic_regression.joblib
│   ├── model_sgd_log.joblib
│   ├── stacking_ensemble.joblib
│   └── benchmark_metrics.json
│
├── tests/                     # pytest tests
│   ├── test_api.py
│   ├── test_models.py
│   └── test_preprocessor.py
│
├── train.py                   # main training pipeline driver
├── retrain_robust_model.py    # robustness retraining
├── run_pipeline.py
├── verify_dataset_model.py  # dataset/model verification
├── cm_audit.py                # confusion-matrix audit
├── test_sample.py
│
├── requirements.txt           # runtime Python deps
├── Dockerfile
├── Procfile
├── render.yaml
├── netlify.toml
└── WELFake_Dataset.csv         # local dataset (git-ignored, ~245 MB)
```

## 3. Tech Stack

| Layer | Technology | Notes |
| --- | --- | --- |
| Frontend | Hand-rolled HTML/CSS/JS (vanilla) | FontAwesome `+` Plus Jakarta Sans from Google |
| Frontend hosting | Netlify / Vercel / static | `_redirects`, `vercel.json` |
| Backend API | **FastAPI** | Pydantic v2 schemas, CORS middleware |
| Server | **uvicorn** | `Procfile`/`render.yaml` boot |
| ML | **scikit-learn** | TF-IDF, classifiers, calibration |
| Deep learning | **PyTorch** | LSTM/attention, CNN+BiLSTM (defined for training) |
| Vectorization | sublinear TF-IDF | 1-2 n-grams, 50k features |
| Ensemble | soft voting + stacking | weights `[0.45, 0.45, 0.10]` |
| Meta | pandas, numpy, joblib | serialization/Persist |
| Testing | **pytest** | `tests/` |

**Label convention:** `0 = Fake News`, `1 = Real News` (matches WELFake dataset definition).

## 4. Data Flow (training)

1. `loader.load_raw_dataset` reads CSV → drop index col → impute → filter empty → cast label.
2. `get_stratified_splits` → train/val/test (70/15/15, seed 42).
3. `TextPreprocessor.transform` → fused text features (title repeat, style).
4. `train.py` vectorizes & trains each model, saves each to `artifacts/`.
5. `best_model.joblib` = classifier with best holdout accuracy; metrics → `benchmark_metrics.json`.

## 5. Serving / Deployment Topology

```
Browser ──► Netlify/Vercel (static site)
              │  fetch(https://ai-powered-fake-news-detection-bcbb.onrender.com)
              ▼
        Render / Railway (FastAPI)
              │  loads artifacts lazily (cached in module global)
              ▼
        artifacts/*.joblib
```

- Frontend **auto-detects** backend URL: localhost/file → `127.0.0.1:8000`, else the Render endpoint.
- Backend lazily loads model once and caches in `model_pipeline` (get_model fallback chain).