# Dataset Study 03: NLP Preprocessing & Tokenization Specification

**Project**: AI-Powered Fake News Detection Using NLP & Deep Learning  
**Dataset**: WELFake (`WELFake_Dataset.csv`)  
**Purpose**: Standardized, deterministic preprocessing and tokenization pipeline for baseline ML, recurrent neural networks, and transformer models.

---

## 1. End-to-End Cleaning Pipeline Architecture

```
Raw CSV Record ('title', 'text')
   │
   ▼
[1] Null & Blank Imputation
   │  - Replace NaN with ""
   │  - Drop records where len(title + text) < 10 chars
   │
   ▼
[2] Publisher Dateline & Artifact Sanitization (Leakage Defense)
   │  - Regex: Remove '^[A-Z\s]+ \(Reuters\) -'
   │  - Regex: Remove ' - Breitbart$', ' \[VIDEO\]$', ' \[TWEET\]$'
   │  - Replace Unicode encoding errors (� -> apostrophe/dash)
   │
   ▼
[3] Text Normalization
   │  - HTML entity unescaping (&amp; -> &, &quot; -> ")
   │  - URL & Email extraction/removal (https?://\S+ -> [URL])
   │  - Twitter/Social handle normalization (@username -> [USER])
   │  - Whitespace compression (\s+ -> " ")
   │
   ▼
[4] Mode-Specific Representation
   ├── Route A: Classical ML (TF-IDF)
   │      - Lowercased, Contraction expanded, Stopwords removed, N-gram (1,3)
   ├── Route B: Deep Learning (BiLSTM / CNN)
   │      - Punctuation-preserved tokenization, Pretrained GloVe/FastText mapping
   └── Route C: Transformer (BERT / RoBERTa / DeBERTa)
          - Structured template: "[CLS] [TITLE] {title} [BODY] {text} [SEP]"
```

---

## 2. Leakage Sanitization Rules (Exact Regular Expressions)

To eliminate wire agency shortcuts, apply the following deterministic regex replacements:

```python
import re
import html

def sanitize_leakage(text: str) -> str:
    if not isinstance(text, str):
        return ""
    # 1. Unescape HTML
    text = html.unescape(text)
    # 2. Strip Reuters / AP Datelines: e.g. "WASHINGTON (Reuters) -", "LONDON (Reuters) -"
    text = re.sub(r'^[A-Z\s,/\.]+\((Reuters|AP|AFP)\)\s*[-—–:]\s*', '', text)
    # 3. Strip trailing agency tags
    text = re.sub(r'\s*[-—–|]\s*(Breitbart|Reuters|The Onion)$', '', text, flags=re.IGNORECASE)
    # 4. Normalize bracketed media tags: [VIDEO], [PHOTOS], [TWEET]
    text = re.sub(r'\[(VIDEO|PHOTOS?|TWEET|AUDIO|WATCH)\]', r'', text, flags=re.IGNORECASE)
    # 5. Normalize URLs and handles
    text = re.sub(r'https?://\S+|www\.\S+', '[URL]', text)
    text = re.sub(r'@\w+', '[USER]', text)
    # 6. Normalize multiple whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text
```

---

## 3. Title + Body Fusion Strategies

| Model Family | Fusion Strategy | Input Format |
| :--- | :--- | :--- |
| **Baseline TF-IDF** | Weighted Concat | `(title + " ") * 2 + text` (doubles title n-gram importance) |
| **BiLSTM / CNN** | Dual-Channel or Concatenated | Fixed 300 words: `title + " | " + text[:280_words]` |
| **Transformers** | Dual-Segment Tokenizer | `tokenizer(title, text, max_length=512, truncation="only_second")` |

> **Transformer Truncation Strategy (`only_second`)**:
> Always retain 100% of the `title` tokens and truncate only the trailing excess of `text`. This ensures the headline's crucial framing is never clipped.

---

## 4. Tokenizer & Embedding Specifications

| Component | Classical NLP | Recurrent / CNN Models | Transformer Models |
| :--- | :--- | :--- | :--- |
| **Tokenizer** | NLTK Regex / Scikit-Learn | Keras Tokenizer / TorchText | `AutoTokenizer` (WordPiece / BPE) |
| **Vocabulary Size**| 25,000 top n-grams (1-3) | 50,000 words | 30,522 (BERT) / 50,265 (RoBERTa) |
| **Embedding Matrix**| Sparse TF-IDF (Sublinear TF) | GloVe 6B 300d / FastText 300d | Pretrained contextual embeddings |
| **Out-Of-Vocab** | Ignored (`min_df=3`) | `<OOV>` index = 1 | `[UNK]` token mapping |
| **Max Sequence Len**| N/A (Sparse vector) | 300 tokens (Pad: `post`) | 512 tokens (Pad: `max_length`) |
