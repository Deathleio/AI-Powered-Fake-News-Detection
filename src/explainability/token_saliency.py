import re
import numpy as np
from typing import List, Dict, Any, Tuple

def extract_tfidf_word_importance(
    text: str,
    pipeline,
    top_k: int = 10
) -> Dict[str, Any]:
    """
    Extracts high-impact words driving classification for a specific sample.
    """
    if hasattr(pipeline, 'vectorizer') and hasattr(pipeline, 'clf'):
        vectorizer = pipeline.vectorizer
        clf = pipeline.clf
    elif hasattr(pipeline, 'named_steps'):
        vectorizer = pipeline.named_steps['tfidf']
        clf = pipeline.named_steps['clf']
    else:
        return {"fake_indicators": [], "real_indicators": []}
    
    X_vec = vectorizer.transform([text])
    feature_names = np.array(vectorizer.get_feature_names_out())
    
    coefs = None
    if hasattr(clf, 'coef_'):
        coefs = clf.coef_[0]
    elif hasattr(clf, 'calibrated_classifiers_'):
        coefs = np.mean([cc.estimator.coef_[0] for cc in clf.calibrated_classifiers_], axis=0)
    
    if coefs is None:
        return {"fake_indicators": [], "real_indicators": []}
    
    row = X_vec.tocoo()
    word_contributions = []
    for idx, val in zip(row.col, row.data):
        score = val * coefs[idx]
        word_contributions.append((feature_names[idx], float(score)))
        
    fake_words = sorted([w for w in word_contributions if w[1] > 0], key=lambda x: x[1], reverse=True)[:top_k]
    real_words = sorted([w for w in word_contributions if w[1] < 0], key=lambda x: x[1])[:top_k]
    
    return {
        "fake_indicators": [{"token": w[0], "weight": round(w[1], 4)} for w in fake_words],
        "real_indicators": [{"token": w[0], "weight": round(abs(w[1]), 4)} for w in real_words]
    }

def generate_highlighted_html(text: str, fake_tokens: List[str], real_tokens: List[str]) -> str:
    html_text = text
    for t in fake_tokens:
        pattern = re.compile(rf'\b({re.escape(t)})\b', re.IGNORECASE)
        html_text = pattern.sub(r'<span style="background-color: #ffcccc; color: #990000; font-weight: bold; padding: 2px 4px; border-radius: 3px;">\1</span>', html_text)
        
    for t in real_tokens:
        pattern = re.compile(rf'\b({re.escape(t)})\b', re.IGNORECASE)
        html_text = pattern.sub(r'<span style="background-color: #ccffcc; color: #006600; font-weight: bold; padding: 2px 4px; border-radius: 3px;">\1</span>', html_text)
        
    return f"<div style='line-height: 1.6; font-size: 15px;'>{html_text}</div>"
