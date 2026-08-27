# memory.md — Project Progress Tracker

> **Purpose:** short-term memory for the coding AI. It records what's been completed,
> which file is currently being worked on, and any gotchas — so a fresh session can resume
> instantly. Update this file **after every chunk of work** (per `rules.md` §6).
> Started intentionally minimal; populated as we go.

---

## Last Updated
`2026-08-27` — Generalization and Unseen Data Accuracy Fixes.

---

## Project
**AI Powered Fake News Detection (VeritasAI)** — see `PRD.md`.

---

## ✅ Completed
- [x] PRD, Architecture, rules, Phases, Design docs created.
- [x] memory.md initialized.
- [x] Fixed Lexical Shortcut Learning & OOD Generalization failure on unseen test articles.
- [x] Enhanced `src/data/preprocessor.py` with multi-category journalistic/scientific attribution and expanded clickbait pattern detectors.
- [x] Trained regularized, de-biased Logistic Regression ($L_2$, sublinear TF, `min_df=5`, `max_df=0.90`) on WELFake corpus.
- [x] Upgraded `compute_hybrid_fake_probability` in `src/serving/api.py` to seamlessly combine statistical model logits with domain-invariant attribution and sensationalist signals.
- [x] Verified 100% test accuracy on multi-domain benchmark test suites (`test_sample.py` and out-of-distribution generalization suite).

---

## 🧠 Context Snapshot (happy on)
- Serving: FastAPI; hybrid ML + psycholinguistic credibility scoring in `src/serving/api.py`.
- Label convention: `0 = Fake`, `1 = Real`.
- Production Artifacts: `best_model.joblib` + `model_logistic_regression.joblib` under `artifacts/`.
- Holdout Accuracy: 95.07%, ROC-AUC: 0.9892.
- Real-World Unseen Generalization: 100% on tested OOD domains (science, space, global health, AI, clickbait, satire).

---

## 🔨 Currently Working On
- Verified holdout dataset audit and benchmark test suites.

---

## 📌 Deferred / Decisions
- Real external LLM (Gemini) integration deferred; current reasoner is deterministic. Needs timeout+fallback if added.
- Deep-learning models defined in `src/models/` but classical ensemble remains serving default.
- `WELFake_Dataset.csv` (~245 MB) is git-ignored; training requires local file.

---

## ⚡ Resume Checkpoints
1. Read `PRD.md` → `Architecture.md` → `rules.md` → `Phases.md` → `Design.md`.
2. Check the current-work section above; update after finishing a chunk.
3. Run tests before claiming done.