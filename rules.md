# Rules.md — Coding Rules & AI Boundaries

> These rules govern ALL code written by the AI (and humans) in this project.
> The AI must honor these boundaries on every edit. Ask/flag before violating.

---

## 1. Core Principles

1. **Label convention is sacred.** `0 = Fake News`, `1 = Real News`. Do NOT flip this anywhere (loaders, model, API, tests). If you think it's inverted, verify with the dataset/audit script, never assume.
2. **Explainability is a feature, not an afterthought.** Every classification path must produce a reason humans can read.
3. **Small, reviewable diffs.** One logical change per edit. Prefer additive changes.
4. **Follow existing structure.** New code lives in the right `src/**` subpackage (data / models / evaluation / explainability / llm_reasoner / serving). No stray root scripts unless they are entrypoints.
5. **Keep it boring & reproducible.** Fit/train scripts must be deterministic (fixed `RANDOM_SEED=42`).
6. **Health over heroics.** Never block the `/health` probe or cold-start model loading behind long offline work.

---

## 2. What to USE (approved)

| Concern | Approved |
| --- | --- |
| Backend | **FastAPI**, **uvicorn**, **pydantic v2**, CORS middleware |
| ML / NLP | **scikit-learn** (TF-IDF, classifiers, calibration), **pandas**, **numpy** |
| Deep learning | **PyTorch** (`torch.nn`, DataLoader) — for LSTM/attention, CNN+BiLSTM |
| Serialization | **joblib** for sklearn pipelines/models |
| Frontend | Vanilla HTML/CSS/JS, **Font Awesome 6**, **Google Fonts (Plus Jakarta Sans)** |
| Tests | **pytest** |
| Serving | **uvicorn**; deployment via Procfile / render.yaml / Dockerfile / netlify.toml |
| Data structures/typing | Python 3.10+ type hints, pydantic `BaseModel` schemas |

## 2.1 Libraries to track — `requirements.txt`

`fastapi`, `uvicorn`, `scikit-learn`, `joblib`, `numpy`, `pandas`, `pydantic`. Add `torch` when the deep architectures are enabled. Keep the file minimal; every entry is deployed to the server.

---

## 3. What to AVOID

- **No unapproved frameworks:** no Django, Flask, React/Vue build tooling, jQuery, Bootstrap for the Netlify frontend (the FastAPI fallback page may use Bootstrap *only* as legacy fallback).
- **No heavy heavyweights in the happy path:** don't import `torch`/`transformers`/`spaCy` at serve time unless truly required and proven faster/accurate.
- **No hard-coded secrets / API keys** in source. Config via env vars (`GEMINI_API_KEY`) only.
- **No silent label flips or magic thresholds** outside a clearly-grepped constant.
- **No reading the huge CSV at import time**; only the loader touches it, and only in training scripts.
- **Don't add new root-level `*.py` scripts** outside the existing entrypoints (`train.py`, `retrain_robust_model.py`, `run_pipeline.py`, `verify_dataset_model.py`) without a reason.
- **No external LLM network calls** in the reasoner without adding timeout + deterministic fallback (see boundaries).

---

## 4. Error Handling Rules

1. **All public endpoints** validate input and return semantic HTTP codes:
   - Empty title+text → `400` with human message.
   - Model artifacts missing → `503` "models still training / not found".
   - Unknown/internal → `500` logged server-side; minimal leak of internals.
2. **Model load failure → graceful fallback chain:** `best_model.joblib` → `model_logistic_regression.joblib` → `model_passive_aggressive.joblib`; raise `503` only if all missing.
3. **Frontend always has a loading + failure state; never a silent hang.** Use AbortController/timeouts and retry for cold-sleep web services.
4. **Predictions are clipped** (`np.clip(..., 0.0001, 0.9999)`) so probability/confidence is stable and in (0,1).
5. **Style feature extraction must not crash on missing punctuation or weird unicode.** Use `re`/string fallbacks; default scores to `0`.
6. **Train scripts fail loudly** on missing dataset (`FileNotFoundError`), missing columns, or malformed labels — don't silently drop the whole table.

---

## 5. Boundaries for the AI (what it may/may NOT change)

**May do freely:**
- Refactor within `src/**` keeping behavior.
- Add/adjust frontend CSS/JS for the current design system.
- Write/extend `tests/`.
- Update `requirements.txt`, config hyperparameters (with a noted rationale).
- Update `memory.md`, `Phases.md`, `PRD.md`, `Architecture.md`, `Design.md` as reality changes.

**Must ask / flag before:**
- Changing the label convention or confidence formula.
- Swapping the core model family (classical ML → deep learning) for serving.
- Adding real external LLM/API calls (Gemini) — needs timeout + fallback design first.
- Adding a new third-party dependency.
- Restructuring deployment (moving to a different hosting topology).

**Never do:**
- Delete or rename the primary dataset, restore from regression, or break the `0/1` label meaning.
- Rewrite the visual identity without going through `Design.md`.
- Overwrite files mid-edit with untracked build outputs or large artifacts.

---

## 6. Working Agreement (for vibe coding)

- **Plan before edits:** state what you'll change, then implement.
- **One logical change per PR/commit.**
- **Run tests before calling done.** `/predict` and `/explain` must return clean JSON.
- Update **`memory.md`** after each completed chunk (see its template).
- If something conflicts with these rules, stop and ask — don't silently override.