# memory.md — Project Progress Tracker

> **Purpose:** short-term memory for the coding AI. It records what's been completed,
> which file is currently being worked on, and any gotchas — so a fresh session can resume
> instantly. Update this file **after every chunk of work** (per `rules.md` §6).
> Started intentionally minimal; populated as we go.

---

## Last Updated
`2026-08-27` — Modernized, Jargon-Free UI Redesign & Enterprise Deployment.

---

## Project
**AI Powered Fake News Detection (VeritasAI)** — see `PRD.md`.

---

## ✅ Completed
- [x] PRD, Architecture, rules, Phases, Design docs created.
- [x] memory.md initialized.
- [x] Fixed Lexical Shortcut Learning & OOD Generalization failure on unseen test articles.
- [x] Created `src/data/download_datasets.py` to automatically download and merge LIAR (10,164 short claims) and CoAID (2,128 healthcare claims) with WELFake into `dataset_study/unified_multidomain_dataset.csv` (75,970 samples).
- [x] Integrated real-time encyclopedic fact-checking knowledge grounding in `src/llm_reasoner/fact_check_agent.py` using Wikipedia Open Search API.
- [x] Retrained regularized multi-domain model with balanced feature weights.
- [x] Created `LLM_CONTINUATION_GUIDE.md` for seamless context handoff across LLM sessions.
- [x] Built URL parsing, publisher domain registry, and claim-by-claim forensic analysis.
- [x] Redesigned UI to be clean, modern, friendly, and jargon-free for non-technical users.
- [x] Pushed to GitHub with automated Vercel & Render cloud deployments.

---

## 🧠 Context Snapshot (happy on)
- Serving: FastAPI; hybrid ML + psycholinguistic credibility + live knowledge grounding in `src/serving/api.py`.
- Label convention: `0 = Fake`, `1 = Real`.
- Frontend: Modern, accessible, consumer-friendly layout with 1-click sample presets and clear visual trust indicators.
- Production Artifacts: `best_model.joblib` + `model_logistic_regression.joblib` under `artifacts/`.
- Handoff Guide: Read `LLM_CONTINUATION_GUIDE.md` for quick start & operational runbook.

---

## 🔨 Currently Working On
- Verified UI and API endpoints across all benchmarks. Ready for end-user testing.

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