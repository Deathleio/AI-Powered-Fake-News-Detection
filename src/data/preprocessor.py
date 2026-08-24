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
