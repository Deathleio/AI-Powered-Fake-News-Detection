import re
import html
import unicodedata
from typing import List, Optional, Tuple
import pandas as pd

def sanitize_wire_leakage(text: str) -> str:
    """
    Strips explicit news wire agency datelines and source watermarks 
    (e.g., 'WASHINGTON (Reuters) -', 'LONDON (AP) --', 'Breitbart')
    to prevent spurious shortcut learning.
    """
    if not isinstance(text, str) or not text:
        return ""
    
    # 1. Unescape HTML entities
    text = html.unescape(text)
    
    # 2. Normalize Unicode (e.g. typographic quotes, accents)
    text = unicodedata.normalize('NFKD', text)
    
    # 3. Strip starting datelines: "CITY (Reuters) - ...", "CITY, State (AP) -- ..."
    text = re.sub(r'^[A-Z\s,/\.\–\-]+\s*\((Reuters|AP|AFP|Bloomberg|CNN)\)\s*[-—–:]*\s*', '', text, flags=re.IGNORECASE)
    
    # 4. Strip trailing news organization stamps
    text = re.sub(r'\s*[-—–|]\s*(Breitbart|Reuters|The Onion|Associated Press)\s*$', '', text, flags=re.IGNORECASE)
    
    # 5. Remove explicit bracketed tags like [VIDEO], [PHOTOS], [TWEET]
    text = re.sub(r'\[\s*(VIDEO|PHOTOS?|TWEET|AUDIO|WATCH|EXCLUSIVE)\s*\]', '', text, flags=re.IGNORECASE)
    
    # 6. Replace URLs and User Mentions
    text = re.sub(r'https?://\S+|www\.\S+', ' [URL] ', text)
    text = re.sub(r'@\w+', ' [USER] ', text)
    
    # 7. Compress whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def fuse_title_body(title: Optional[str], text: Optional[str], title_repeat: int = 1) -> str:
    """
    Fuses title and body into a structured representation with optional title weighting.
    """
    clean_title = sanitize_wire_leakage(str(title) if pd.notna(title) else "")
    clean_body = sanitize_wire_leakage(str(text) if pd.notna(text) else "")
    
    if not clean_title and not clean_body:
        return ""
    
    if not clean_title:
        return clean_body
    if not clean_body:
        return clean_title
        
    weighted_title = " ".join([clean_title] * title_repeat)
    return f"{weighted_title} . {clean_body}"

def extract_stylistic_features(title: Optional[str], text: Optional[str]) -> dict:
    """
    Extracts structural, capitalization, punctuation, and journalistic attribution signals
    prior to text normalization and lowercasing.
    """
    t = str(title) if pd.notna(title) and title else ""
    b = str(text) if pd.notna(text) and text else ""
    full_raw = f"{t} {b}".strip()
    
    if not full_raw:
        return {
            "caps_ratio": 0.0,
            "title_caps_ratio": 0.0,
            "is_all_caps_title": False,
            "is_all_caps_body": False,
            "exclamation_density": 0.0,
            "sensational_keywords": [],
            "sensational_score": 0.0,
            "attribution_score": 0.0,
            "stylistic_fake_risk": 0.0
        }

    # 1. Capitalization ratios
    letters_title = [c for c in t if c.isalpha()]
    letters_body = [c for c in b if c.isalpha()]
    letters_total = [c for c in full_raw if c.isalpha()]
    
    title_caps_ratio = sum(1 for c in letters_title if c.isupper()) / max(1, len(letters_title))
    body_caps_ratio = sum(1 for c in letters_body if c.isupper()) / max(1, len(letters_body))
    total_caps_ratio = sum(1 for c in letters_total if c.isupper()) / max(1, len(letters_total))
    
    is_all_caps_title = len(letters_title) >= 6 and title_caps_ratio >= 0.65
    is_all_caps_body = len(letters_body) >= 15 and body_caps_ratio >= 0.65

    # 2. Exclamation & question mark density
    exclamations = full_raw.count('!') + full_raw.count('?')
    exclamation_density = exclamations / max(1, len(full_raw.split()))

    # 3. Sensationalist & clickbait pattern matcher
    sensational_patterns = [
        r'\b(world is on fire|on fire)\b',
        r'\b(shocking|bombshell|unbelievable|explosive|mind-blowing)\b',
        r'\b(secret plot|globalist plot|conspiracy|covert scheme|cover-?up)\b',
        r'\b(mainstream media refuses|media won\'?t show|they don\'?t want you to see)\b',
        r'\b(breaking news|must see|watch before deleted|viral video)\b',
        r'\b(confiscate savings|martial law|arrest warrant leaked)\b',
        r'\b(dancing with|bizarre ritual|secretly meeting)\b'
    ]
    detected_sensational = []
    for pattern in sensational_patterns:
        matches = re.findall(pattern, full_raw, flags=re.IGNORECASE)
        if matches:
            detected_sensational.extend(matches if isinstance(matches[0], str) else [m[0] for m in matches])
    
    # 4. Legitimate journalistic attribution patterns
    attribution_patterns = [
        r'\b(according to|spokesperson said|officials confirmed|in a statement|in an interview)\b',
        r'\b(press conference|reported on|cited sources|internal memo|department of)\b',
        r'\b(told reporters|analysts noted|preliminary data showed|reuters|associated press)\b'
    ]
    detected_attributions = []
    for pattern in attribution_patterns:
        matches = re.findall(pattern, full_raw, flags=re.IGNORECASE)
        if matches:
            detected_attributions.extend(matches)
            
    attribution_score = min(1.0, len(detected_attributions) * 0.35)

    # 5. Composite Stylistic Fake Risk Score [0.0 to 1.0]
    risk = 0.0
    if is_all_caps_title:
        risk += 0.35
    if is_all_caps_body:
        risk += 0.35
    elif total_caps_ratio > 0.40:
        risk += 0.20
        
    if detected_sensational:
        risk += min(0.40, len(detected_sensational) * 0.20)
    if exclamation_density > 0.05:
        risk += 0.20
        
    # Subtract attribution evidence
    risk = max(0.0, min(1.0, risk - attribution_score * 0.4))

    return {
        "caps_ratio": round(total_caps_ratio, 4),
        "title_caps_ratio": round(title_caps_ratio, 4),
        "is_all_caps_title": bool(is_all_caps_title),
        "is_all_caps_body": bool(is_all_caps_body),
        "exclamation_density": round(exclamation_density, 4),
        "sensational_keywords": list(set(detected_sensational)),
        "sensational_score": round(min(1.0, len(detected_sensational) * 0.25), 2),
        "attribution_score": round(attribution_score, 2),
        "stylistic_fake_risk": round(risk, 4)
    }

class TextPreprocessor:
    """
    High-performance text cleaning and feature transformer for Fake News classification.
    """
    def __init__(self, title_repeat: int = 2):
        self.title_repeat = title_repeat

    def fit(self, X, y=None):
        return self

    def transform(self, df: pd.DataFrame) -> List[str]:
        titles = df['title'].fillna('') if 'title' in df.columns else [''] * len(df)
        texts = df['text'].fillna('') if 'text' in df.columns else [''] * len(df)
        
        fused = [
            fuse_title_body(t, b, title_repeat=self.title_repeat)
            for t, b in zip(titles, texts)
        ]
        return fused

