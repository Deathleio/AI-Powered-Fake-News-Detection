# Coding Standards & Boundaries

These rules govern code written for VeritasAI:

## 1. Core Principles
1. **Label convention is sacred:** `0 = Fake News`, `1 = Real News`. Never invert or change this in data loaders, models, APIs, or tests.
2. **Explainability is mandatory:** Every prediction path must provide interpretable token saliency, claim tags, and human-readable reasoning.
3. **Reproducibility:** Fit/train scripts must remain deterministic (`RANDOM_SEED=42`).
4. **Project Structure:**
   - Production source code lives under `src/**`.
   - Serialized models and weights live in `artifacts/`.
   - Test suites live in `tests/`.
   - Documentation lives in `docs/` and `README.md`.
5. **Serving Latency & Resilience:** Keep the default serving pipeline lightweight (<50MB RAM, <5ms latency). Always provide graceful fallback chains when loading models.
