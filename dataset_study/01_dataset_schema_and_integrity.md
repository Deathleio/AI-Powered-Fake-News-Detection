# Dataset Study 01: Dataset Schema & Data Integrity Specification

**Project**: AI-Powered Fake News Detection Using NLP & Deep Learning  
**Dataset**: WELFake (`WELFake_Dataset.csv`)  
**Target Audience**: Autonomous Agents, LLMs, and Machine Learning Engineers  
**Goal**: Token-efficient, exhaustive specification of schema, data types, integrity boundaries, null-handling protocols, and class semantics without loading raw data.

---

## 1. Storage & File Profile
- **File Name**: `WELFake_Dataset.csv`
- **File Size**: 245,086,152 bytes (~245.08 MB)
- **Total Record Count**: 72,134 rows
- **Total Columns**: 4
- **File Encoding**: UTF-8 with standard Windows/Unix line terminators
- **Header Row**: Present on line 1 (`Unnamed: 0,title,text,label`)

---

## 2. Schema Definition & Data Types

| Column Name | Storage Type | Logical Data Type | Role | Null Count | Null % | Description & Constraints |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `Unnamed: 0` | `int64` | Primary Key / Index | Identifier | 0 | 0.00% | Integer index ranging sequentially from `0` to `72133`. Non-null, strictly monotonic. |
| `title` | `object` (str) | Text / Headline | Input Feature | 558 | 0.77% | Headline of the news article. Min length: 0 chars, Max length: 72 words (~450 chars). |
| `text` | `object` (str) | Text / Body | Input Feature | 39 | 0.05% | Full article text body. Min length: 0 chars, Max length: 24,234 words (~150,000 chars). |
| `label` | `int64` | Binary Target | Ground Truth | 0 | 0.00% | Discrete binary target: `0` (Real/Mainstream News), `1` (Fake/Hyperpartisan News). |

---

## 3. Ground Truth Semantics & Class Mapping

The WELFake corpus merges four distinct benchmark news datasets:
1. Kaggle Fake News Dataset (train.csv)
2. McIntire Fake News Dataset
3. Reuters News Articles Collection
4. BuzzFeed Political News Dataset

### Class Distribution Profile:
```
Total Rows: 72,134
├── Class 1 (Fake / Sensational / Hyperpartisan): 37,106 rows (51.44%)
└── Class 0 (Real / Verified / Mainstream News):  35,028 rows (48.56%)
```
- **Imbalance Ratio**: `1.059 : 1.0` (virtually balanced).
- **Target Weighting**: Uniform class weighting (`weight=[1.0, 1.0]`) or standard Binary Cross Entropy / Log-Loss is suitable without synthetic oversampling (SMOTE is not required).

---

## 4. Integrity Constraints & Preprocessing Rules for Subagents

When subagents ingest or batch this dataset, they must enforce the following deterministic integrity pipeline:

```
Raw Row Ingestion
   │
   ├── Check: Are BOTH 'title' and 'text' null/empty? ───► YES: DROP ROW (33 records)
   │
   ├── Check: Is 'title' null/NaN? ──────────────────────► YES: IMPUTE WITH "" (525 records)
   │
   ├── Check: Is 'text' null/NaN? ───────────────────────► YES: IMPUTE WITH "" (6 records)
   │
   └── Feature Fusion: Concatenate `title` + " " + `text` ─► Clean Text Pipeline
```

### Critical Data Cleaning Directives:
1. **Drop Invalid Rows**: Filter out records where `(title.isna() | title.str.strip() == "") & (text.isna() | text.str.strip() == "")`.
2. **Handle Special Characters & Typographic Quotes**: UTF-8 artifacts (e.g. `�`, `â€™`, `â€œ`, smart quotes, hyphens) must be normalized to standard ASCII punctuation before subword tokenization.
3. **Strip Source Dateline Shortcuts**: Over 21,637 rows in Class 0 contain explicit news wire markers like `(Reuters) -` or `BRUSSELS (Reuters)`. Models must have these stripped during preprocessing to avoid trivial lexical shortcut learning.

---

## 5. Duplicate & Overlap Profile

- **Exact Duplicate Rows across all 4 columns**: `0`
- **Duplicate `text` bodies**: `9,415` rows
- **Duplicate `title` headlines**: `9,786` rows

### Leakage Prevention Protocol:
When splitting data into Train, Validation, and Test partitions:
- Do **NOT** use naive random splitting.
- Apply **GroupKFold** or hash-based deduplication on normalized text to guarantee that syndicated news variants do not appear in both Train and Test splits simultaneously.

---

## 6. Token Budget Guidelines for LLM Agents

To prevent context exhaustion in multi-agent workflows:
- **Never feed raw CSV text into LLM prompts.**
- **Pass structured metadata, batch indices, or top-k representative samples (max 300 words per snippet).**
- Maximum context consumption per agent analysis step: `< 1,500 tokens`.
