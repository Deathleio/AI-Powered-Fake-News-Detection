# memory.md — Project Progress Tracker

> **Purpose:** short-term memory for the coding AI. It records what's been completed,
> which file is currently being worked on, and any gotchas — so a fresh session can resume
> instantly. Update this file **after every chunk of work** (per `rules.md` §6).
> Started intentionally minimal; populated as we go.

---

## Last Updated
`2026-08-25` — initial baseline snapshot.

---

## Project
**AI Powered Fake News Detection (VeritasAI)** — see `PRD.md`.

---

## ✅ Completed
- [x] PRD, Architecture, rules, Phases, Design docs created.
- [x] memory.md initialized (this file).
- *(baseline code already present in repo — see below)*

*(Detailed "what's done" will be added as work is performed in later phases.)*

---

## 🧠 Context Snapshot (happy on)
- Serving: FastAPI; model fallback chain in `src/serving/api.py`.
- Label convention: `0 = Fake`, `1 = Real`.
- Artifacts: `best_model.joblib` + siblings under `artifacts/`.
- Frontend token-driven (`frontend/style.css`), warm editorial theme.

---

## 🔨 Currently Working On
- **_None yet_ — docs scaffolding only.**
- Next planned work: see `Phases.md` exit criteria (start Phase 1).

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