# Phases.md — Delivery Roadmap

> Baseline: a functional v1 already exists (classical ML + explainability + UI + API).
> Phases below implement the **current roadmap** iteratively.

---

## Phase 0 — Baseline Audit (current state)

**Goal:** lock down what already works before extending.
- [x] FastAPI backend (`/health`, `/predict`, `/explain`).
- [x] TF-IDF + 3 calibrated baselines + stacking ensemble (benchmark ~96–97%).
- [x] Explainability: token saliency + highlighted HTML.
- [x] Rule-based reasoner rationale.
- [x] Responsive static frontend (VeritasAI).
- [x] `render.yaml` / `Dockerfile` / `netlify.toml` / `vercel.json` / `Procfile`.
- [ ] Confirm CI-friendly test pass: `pytest`.

**Exit criteria:** all tests green; `/health` responds; doc files (PRD/Architecture/rules/Phases/Design/memory) present.

---

## Phase 1 — Robustness & Anti-Bias Hardening
- [ ] Reinforce entity-neutralization preprocessing (mask named entities / names of politicians).
- [ ] Add sensationalism/all-caps/attribution features to training, not just serving.
- [ ] Verify generalization on out-of-domain examples using `verify_dataset_model.py`.
- [ ] Confusion-matrix audit (`cm_audit.py`) + error-case review.
- [ ] Retrain robust variant (`retrain_robust_model.py`) and compare benchmark.

**Exit:** improved macro-F1 and reduced entity shortcut reliance, benchmark in artifacts updated.

---

## Phase 2 — Deep Learning Upgrade (experimental, optional)
- [ ] Enable LSTM-attention and CNN+BiLSTM training path (`src/models/*`).
- [ ] Compare vs classical ensemble on same holdout.
- [ ] Keep the best-serving performer in `best_model.joblib`.

**Exit:** decision record on whether deep model replaces or complements classical ensemble.

---

## Phase 3 — User Experience & Feedback
- [ ] History/analytics: keep last N investigations in localStorage.
- [ ] Copy-to-clipboard for the explained report.
- [ ] Loading skeleton + clearer cold-start messaging.
- [ ] Accessibility pass (keyboard focus, ARIA, contrast).
- [ ] Empty/error empty states.

**Exit:** demo flow polished; metric shown is honest (confidence + probability).

---

## Phase 4 — Persistence & Insights (nice-to-have)
- [ ] Optional lightweight backend storage for audit trail.
- [ ] Aggregate dashboards (recent scans, category distribution).
- [ ] Auth for admin dashboard (future; HTTP-pie minimal).

**Exit:** recorded as future; not required for v1.

---

## Phase 5 — Production Hardening & Docs
- [ ] Load tests + graceful degradation doc.
- [ ] Docker image verification.
- [ ] Update `Deployment.md` / `DEPLOYMENT_GUIDE.md` with any new flows.
- [ ] Security review (no secrets, CORS scope, rate-limiting consideration).

**Exit:** deployable, documented, testable release.

---

### Status legend
`[x]` done · `[ ]` todo · `[~]` in progress (track in `memory.md`).