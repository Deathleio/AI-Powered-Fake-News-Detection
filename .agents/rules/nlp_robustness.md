---
description: Robustness standards for NLP text classification, preventing entity shortcut learning and lexical overfitting.
globs: [src/**/*.py, retrain*.py, train.py]
---

# NLP Classification Robustness & Bias Prevention Standards

When designing, preprocessing, training, or serving NLP text classification models:

1. **Entity Neutralization & Confounder Mitigation**:
   - Do not rely solely on bag-of-words/n-grams that over-index on named entities (e.g. politician names, country names, wire stamps).
   - Apply sublinear term scaling (sublinear_tf=True), strong L2 regularization, or entity-masking/stopwords where applicable.

2. **Stylistic & Meta-Feature Extraction**:
   - Preserve or explicitly extract stylistic signals:
     - All-caps ratio (sensationalist shouting detector)
     - Punctuation extremity (multiple exclamation marks, question marks)
     - Sentiment extremity / sensationalism lexicon matches (e.g., SHOCKING, BOMBSHELL, ON FIRE, UNBELIEVABLE)
     - Quotation & attribution density (e.g., 'according to', 'spokesperson said', 'confirmed')

3. **Hybrid Verification Guardrails**:
   - Pure statistical n-gram probabilities must be calibrated against heuristic stylistic checks and LLM semantic plausibility checks before serving final confidence scores.
   - Flag inputs with extreme all-caps or sensationalist syntax as elevated fake/clickbait risk.
