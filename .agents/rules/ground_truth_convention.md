---
description: " Strict Ground Truth and Semantic Polarity Mapping Rule for Fake News Detection\
globs: \**/*.py\
---

# Semantic Ground Truth Mapping & Polarity Standard

## 1. Ground Truth Semantics
- Target Value 1: REAL / AUTHENTIC NEWS (e.g. accredited wire reports from Reuters/AP, formal political profiles).
- Target Value 0: FAKE / DISINFORMATION NEWS (e.g. clickbait, unverified conspiracy claims, satirical absurdities like 'trump is eating children for lunch').

## 2. Ingestion Protocol for WELFake_Dataset.csv
- In raw Kaggle WELFake dataset:
 - Raw label = 0 corresponds to genuine Reuters wire articles (Real).
 - Raw label = 1 corresponds to clickbait/fake news articles (Fake).
- Mandatory Transformation in loader.py:
 df['label'] = 1 - df['label'].astype(int)
 This ensures:
 - Genuine Reuters news -> 1 (Real News)
 - Clickbait / Disinformation -> 0 (Fake News)

## 3. Serving & Probability Output Protocol
- proba_fake = float(model.predict_proba([fused_text])[0, 0]) (Class 0 = Fake)
- proba_real = float(model.predict_proba([fused_text])[0, 1]) (Class 1 = Real)
- is_fake = proba_fake >= 0.5
- verdict = 'Fake / Disinformation News' if is_fake else 'Real / Authentic News'
