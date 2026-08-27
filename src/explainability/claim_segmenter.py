import re
from typing import List, Dict, Any

def segment_and_analyze_claims(title: str, text: str) -> List[Dict[str, Any]]:
    """
    Segments an article into atomic sentence-level claims and assigns:
      - Claim Type: ATTRIBUTED_QUOTE, EMPIRICAL_DATA, SENSATIONAL_TRIGGER, OBJECTIVE_STATEMENT, UNBACKED_ASSERTION
      - Veracity Risk: Low / Medium / High
      - Evidence / Attribution snippet
    """
    full_text = f"{title}. {text}".strip() if title else (text or "").strip()
    if not full_text:
        return []
        
    # Split by sentence boundaries
    raw_sentences = re.split(r'(?<=[.!?])\s+', full_text)
    sentences = [s.strip() for s in raw_sentences if len(s.strip()) > 15]
    
    claims = []
    for idx, sentence in enumerate(sentences[:10], 1):
        s_lower = sentence.lower()
        
        # 1. Attributed quote or cited authority
        is_attributed = bool(re.search(
            r'\b(according to|spokesperson said|officials confirmed|in a statement|told reporters|published in|researchers found|scientists|study by)\b',
            s_lower
        ))
        
        # 2. Sensational / hyperbole / clickbait
        is_sensational = bool(re.search(
            r'\b(shocking|bombshell|unbelievable|explosive|secret plot|miracle cure|cures all|world is on fire|must see|watch before deleted)\b',
            s_lower
        )) or (sum(1 for c in sentence if c.isupper()) / max(1, len(sentence)) > 0.45)
        
        # 3. Quantitative / empirical claim
        is_empirical = bool(re.search(r'\b(\d+(\.\d+)?%|\$\d+|\b\d{4}\b|\bpercent\b)\b', s_lower))
        
        if is_sensational:
            category = "High-Risk Sensational Claim"
            risk_level = "High Risk"
            tag_class = "risk-high"
            note = "Contains alarmist hyperbole or unsupported emotional clickbait framing."
        elif is_attributed:
            category = "Verified Sourced Statement"
            risk_level = "Low Risk"
            tag_class = "risk-low"
            note = "Directly references institutional or journalistic source attribution."
        elif is_empirical:
            category = "Empirical Data Point"
            risk_level = "Medium Risk"
            tag_class = "risk-medium"
            note = "Cites statistical figures; corroborate with primary documentation."
        else:
            category = "Contextual Narrative"
            risk_level = "Neutral"
            tag_class = "risk-neutral"
            note = "Standard descriptive narrative framing."
            
        claims.append({
            "claim_id": idx,
            "text": sentence,
            "category": category,
            "risk_level": risk_level,
            "tag_class": tag_class,
            "note": note
        })
        
    return claims
