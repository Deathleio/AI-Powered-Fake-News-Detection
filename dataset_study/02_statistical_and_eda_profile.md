# Dataset Study 02: Statistical & Exploratory Data Analysis Profile

**Project**: AI-Powered Fake News Detection Using NLP & Deep Learning  
**Dataset**: WELFake (`WELFake_Dataset.csv`)  
**Purpose**: Quantitative and statistical benchmark of text distributions, vocabulary diversity, stylistic markers, and feature correlations.

---

## 1. Corpus-Level Summary Metrics

| Metric | Overall Dataset | Class 0 (Real News) | Class 1 (Fake News) | Divergence / Significance |
| :--- | :--- | :--- | :--- | :--- |
| **Total Samples** | 72,134 | 35,028 (48.56%) | 37,106 (51.44%) | Balanced |
| **Avg Title Word Count** | 12.17 words | 11.05 words | 13.24 words | Fake headlines are ~20% longer |
| **Std Title Word Count** | 4.26 words | 2.85 words | 5.02 words | Fake headlines exhibit higher variance |
| **Max Title Word Count** | 72 words | 30 words | 72 words | Outlier titles are exclusively fake |
| **Avg Body Word Count** | 540.55 words | 577.62 words | 505.56 words | Real news articles are longer on avg |
| **Median Body Word Count**| 398.00 words | 427.00 words | 369.00 words | Real news median is +15.7% larger |
| **Avg Title Caps Ratio** | 16.50% | 8.56% | **23.99%** | **Fake titles have ~2.8x uppercase letters** |
| **Avg Body Caps Ratio** | 4.18% | 3.82% | 4.51% | Fake bodies have higher exclamation/shouting |

---

## 2. Text Length Quantile Distribution

Understanding sequence lengths is critical for setting `max_seq_len` for Deep Learning (BiLSTM/CNN/BERT):

### Headline (`title`) Word Length Quantiles:
- **Min**: 0 words
- **25th Percentile**: 9 words
- **50th Percentile (Median)**: 12 words
- **75th Percentile**: 14 words
- **90th Percentile**: 17 words
- **99th Percentile**: 25 words
- **Max**: 72 words

### Article Body (`text`) Word Length Quantiles:
- **Min**: 0 words
- **25th Percentile**: 227 words
- **50th Percentile (Median)**: 398 words
- **75th Percentile**: 667 words
- **90th Percentile**: 1,102 words
- **95th Percentile**: 1,415 words
- **99th Percentile**: 2,767 words
- **Max**: 24,234 words

```
Cumulative Coverage by Sequence Length (Words):
├── 256 words:  ~32% of full articles covered without truncation
├── 384 words:  ~48% of full articles covered without truncation
├── 512 words:  ~65% of full articles covered without truncation (Optimal Transformer limit)
└── 1024 words: ~88% of full articles covered without truncation (Optimal Hierarchical / Longformer)
```

---

## 3. Stylistic & Lexical Signal Markers

### A. Title Capitalization & Sensationalism (Clickbait Signature)
- **Class 1 (Fake)**: Heavy usage of FULL-CAPS words (`BREAKING`, `UNBELIEVABLE`, `SHOCKING`, `EXPOSED`, `WATCH`, `MUST SEE`, `[VIDEO]`, `[TWEET]`).
- **Class 0 (Real)**: Follows AP / Reuters standard headline casing rules (Title Case or Sentence Case) with low uppercase density (< 10%).

### B. Source & Agency Leakage Signals
- **Class 0 (Real)** contains `21,637` instances of the string `Reuters` (61.8% of all real news).
- **Class 0 (Real)** contains `2,355` instances of `Breitbart` in titles.
- **Class 1 (Fake)** contains only `597` mentions of `Reuters` and `46` of `Breitbart`.
- **Implication**: Raw uncleaned text will cause models to achieve artificially high accuracy (>98%) simply by memorizing the word `Reuters`. **Preprocessing must sanitize publisher datelines.**

### C. Sentiment & Punctuation Polarities
- **Class 1 (Fake)**: High frequency of multiple punctuation marks (`???`, `!!!`, `?!`, `...`), emotive adjectives (`corrupt`, `evil`, `insane`, `traitor`, `disaster`).
- **Class 0 (Real)**: Formal journalistic tone, high density of attribution verbs (`said`, `stated`, `reported`, `according to`).

---

## 4. Recommendations for Architecture Design

1. **Dual-Feature Ingestion**: Process both `title` and `text`. The title provides high-density clickbait/stylistic signals; the body provides deep factual/semantic context.
2. **Truncation Cutoff**: Set Transformer max token length to **512 tokens**. Setting title max length to 64 tokens and body max length to 448 tokens captures over 99% of title information and the critical opening paragraphs of articles.
3. **Custom Handcrafted Features**: Augment deep embeddings with engineered stylistic features (caps ratio, punctuation density, sentiment polarity, readability scores).
