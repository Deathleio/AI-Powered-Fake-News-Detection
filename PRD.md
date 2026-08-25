# PRD — AI Powered Fake News Detection

> **Project codename:** VeritasAI
> **Document owner:** AI-assisted development (vibe coding)
> **Status:** Baseline for current implementation (existing codebase snapshot)

---

## 1. Overview

**What we are building:**

An **AI-powered fake news detection system** that classifies news articles (headline + body text) as **Real** or **Fake**, and explains *why* the model made that decision. It combines:

- Classical ML text classification over TF-IDF features.
- Deep-learning style architectures defined for the same task (BiLSTM/CNN).
- **Explainability** — token-level saliency, highlighted text, reason chips.
- A lightweight **LLM-style reasoning layer** that synthesizes a human-readable rationale.
- A modern, mobile-first **web dashboard** to submit articles and view results.

**One-line summary:** Paste or load a news headline + body and get a verdict, a confidence score, the top linguistic signals, and a plain-English explanation.

---

## 2. Problem Statement

Fake news and disinformation spread fast. Readers and researchers need a quick, transparent way to judge the credibility of an article. "Black box" classifiers aren't trustworthy — users need to see *why* something was flagged. This project builds a **high-accuracy plus explainable** solution rather than a disclaimer-style detector.

---

## 3. Goals & Non-Goals

**Goals:**
- Binary classification: **0 = Fake News**, **1 = Real News** (single, consistent convention).
- High measured accuracy on the WELFake dataset (currently benchmarked at ~96–97%).
- Explainable output: saliency tokens, confidence meter, highlighted snippet, rationale.
- Real-time API with production-ready endpoints (`/health`, `/predict`, `/explain`).
- Responsive frontend that works on desktop + mobile.

**Non-Goals:**
- Claiming absolute "truth" of a story — output is a *statistical probability plus reasoning aid*.
- Multi-class/multi-label classification, OCR, audio/video, or live social-media ingestion (future candidate, not now).
- Persistent user accounts / history (future candidate).
- Embargoed/branded content policy — bias guardrails exist but are not a certification.

---

## 4. Target Users

| Persona | What they need | Priority |
| --- | --- | --- |
| **General public / readers** | Paste an article, get a clear "Real vs Fake" verdict + confidence, understand why. | High |
| **Journalists / researchers** | Explainable flags: top trigger tokens, stylistic cues, cited/annotated text. | High |
| **Students of NLP/ML** | Inspect model reasoning, saliency, and architecture blueprints. | Medium |

**Assumptions:** frontend users are non-technical; only developers touch the Python backend, training scripts, and API.

---

## 5. Key Features

**Feature list (MVP / implemented baseline):**

1. **Article Input**
   - Single headline (title) field + article body textarea.
   - Preset "Quick Sample" buttons (Real and Fake examples) for instant demo.

2. **Classification Core**
   - TF-IDF vectorization (sublinear TF, 1–2 n-grams, up to 50k features).
   - Three calibrated baselines: **Passive-Aggressive**, **Logistic Regression**, **SGD Log-Loss**.
   - **Stacking meta-ensemble** combining all three base models.

3. **Explainability**
   - Token saliency extraction (top fake/real indicators with weights).
   - Annotated/highlighted text snippet of the input.
   - Chip UI listing top fake vs real indicators.

4. **LLM-style Reasoning**
   - Structured rationale: all-caps detection, sensationalist keywords, attribution/quoting density, plain-English rationale.

5. **Serving / API**
   - `GET /health` — liveness.
   - `POST /predict` — verdict, fake_probability, confidence_percentage, is_fake.
   - `POST /explain` — everything from `/predict` plus indicators, highlighted HTML, llm_reasoning.
   - CORS enabled for cross-origin frontends.

6. **Analytics / History (future)** — request logging, batch analysis, charts (see Phases.md).

---

## 6. Functional Requirements (FR)

| ID | Requirement |
| --- | --- |
| FR-1 | User can submit a title and body. |
| FR-2 | System returns verdict (`Fake News` / `Real News`), fake probability, and confidence % (0–100). |
| FR-3 | `/predict` returns verdict only; `/explain` returns full analysis. |
| FR-4 | Empty title+body → HTTP 400 with clear message. |
| FR-5 | Model missing (not trained) → HTTP 503 "models still training". |
| FR-6 | Model fallback chain: best_model → logistic regression → passive-aggressive. |
| FR-7 | Frontend endpoint health indicator + retry logic (cold-sleep wake). |
| FR-8 | Token-level explanation chips shown for either fake or real side. |
| FR-9 | 4 preset sample articles (2 real, 2 fake) for demo. |

---

## 7. Non-Functional Requirements

| Category | Requirement |
| --- | --- |
| Accuracy | ≥ 90% holdout accuracy target; currently benchmarked ~96–97%. |
| Latency | `/predict` fast on typical input; model loaded lazily and cached in memory. |
| Availability | Stateless API, deployable to Render/Railway; frontend to Netlify/Vercel. |
| Security | No hard-coded secrets; `GEMINI_API_KEY` via env only when used. |
| Accessibility | Keyboard-usable, high contrast, mobile-friendly layout. |
| Reliability | Graceful model-load fallback chain; frontend retry logic. |
| Maintainability | Modular Python package (`src/**`) with unit tests. |

---

## 8. Success Metrics

- Holdout test accuracy / macro-F1 / ROC-AUC from `artifacts/benchmark_metrics.json`.
- Round-trip API response time.
- % of predictions surfaced with rationale + highlighted text.
- Demo usability: "load sample → get explanation" in under 30s.

---

## 9. Out of Scope (v1)

- Live URL ingestion / scraper.
- Multi-language support.
- Persistent user accounts / history.
- Real external LLM calls (current reasoner is a deterministic rule-based synthesizer; optional later).
- Model retraining UI.

---

## 10. Risks & Assumptions

- **Reasoner is deterministic** (no network call). If a real LLM is integrated later, wrap with timeouts + deterministic fallback.
- **Dataset size:** WELFake CSV (~245 MB) is git-ignored; training needs the file present locally.
- **Entity overfitting:** mitigated via sublinear TF-IDF + stylistic features (see `rules.md` and `.agents/rules/nlp_robustness.md`).
- **Label convention:** strict `0=fake, 1=real` is kept consistent across loaders, model, and API.