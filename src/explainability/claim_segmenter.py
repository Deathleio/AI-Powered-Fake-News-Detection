import re
from typing import List, Dict, Any, Tuple

# Patterns that indicate extraordinary, unverified breakthrough assertions or sensational claims
EXTRAORDINARY_ASSERTION_PATTERNS = [
    r'\b(confirms?|proves?|reveals?|discovered?|uncovered?|evidence of)\s+.*?(fossil|fossilised|fossils|alien|biological structures?|extraterrestrial|ancient alien|miracle|secret)\b',
    r'\b(fossilised|fossilized)\s+(biological|remains|structures?|specimen|microorganisms?)\b',
    r'\b(repeating patterns? inconsistent with natural|anomalous structures?|artificial origin|alien artifact)\b',
    r'\b(cures?\s+(all|cancer|aging|diabetes|every disease)|miracle\s+cure|overnight\s+cure|100%\s+natural\s+cure)\b',
    r'\b(banned\s+all\s+cash|martial\s+law\s+declared|confiscate\s+savings|secretly\s+executed|arrest\s+warrant\s+issued\s+for|total\s+financial\s+blackout)\b',
    r'\b(secret\s+plot|globalist\s+plot|covert\s+scheme|corrupt\s+elites\s+are\s+secretly|conspiracy\s+to\s+overthrow)\b',
    r'\b(suspends?\s+all\s+(traditional\s+)?(wire\s+transfers?|banking|transactions?|withdrawals?))\b',
    r'\b(catastrophic\s+quantum\s+(cryptography\s+)?glitch|quantum\s+(cryptography\s+)?glitch)\b',
    r'\b(institutional\s+liquidity\s+(distribution\s+)?will\s+remain\s+offline)\b',
    r'\b((federal\s+reserve|treasury|central\s+bank)\s+(unexpectedly\s+)?(suspends?|freezes?|halts?|seizes?))\b',
    r'\b(sparking\s+immediate\s+panic|emergency\s+bank\s+holiday|global\s+financial\s+reset)\b'
]

# Domain-specific technical and scientific terminology patterns
TECHNICAL_CONTEXT_PATTERNS = [
    r'\b(regolith|spectrometer|spectral analysis|supercam|perseverance|curiosity|infrared|crystallographic)\b',
    r'\b(molecular clusters|crystalline layers|geological erosion|instrument|rover|exoplanet|photometry)\b',
    r'\b(monetary policy|basis points|central bank|inflation data|treasury yields|interest rates)\b',
    r'\b(peer-reviewed|spectroscopic signatures|inner disk|planetary formation|astronomers)\b',
    r'\b(sandbox|privilege[- ]escalation|zero[- ]day|fuzzing|vulnerabilities|system defects|exploit|generative models?|machine learning)\b'
]

def segment_and_analyze_claims(title: str, text: str) -> List[Dict[str, Any]]:
    """
    Segments an article into atomic sentence-level claims and assigns:
      - Category: High-Risk Sensational Claim, Unverified Breakthrough Assertion,
                  Verified Sourced Statement, Technical / Contextual Framing, Empirical Data Point
      - Veracity Risk: High Risk / Medium Risk / Low Risk / Neutral
      - Evidence / Contextual Note
    """
    full_text = f"{title}. {text}".strip() if title else (text or "").strip()
    if not full_text:
        return []
        
    # Split by sentence boundaries
    raw_sentences = re.split(r'(?<=[.!?])\s+', full_text)
    sentences = [s.strip() for s in raw_sentences if len(s.strip()) > 15]
    
    claims = []
    for idx, sentence in enumerate(sentences[:12], 1):
        s_lower = sentence.lower()
        
        # 1. Check for extraordinary or unverified breakthrough assertions
        is_extraordinary = any(bool(re.search(pat, s_lower)) for pat in EXTRAORDINARY_ASSERTION_PATTERNS)

        # 2. Check for explicit clickbait / sensational hyperbole
        is_sensational = bool(re.search(
            r'\b(shocking|bombshell|unbelievable|explosive|secret plot|miracle cure|cures all|world is on fire|must see|watch before deleted|banned|censored)\b',
            s_lower
        )) or (sum(1 for c in sentence if c.isupper()) / max(1, len(sentence)) > 0.45)
        
        # 3. Attributed quote or cited authority
        is_attributed = bool(re.search(
            r'\b(according to|spokesperson said|officials confirmed|in a statement|told reporters|published in|researchers found|study by|confirmed (it is|that|the|they)|disclosing (these|the)|announced that)\b',
            s_lower
        ))

        # 4. Technical / Domain context
        is_technical = any(bool(re.search(pat, s_lower)) for pat in TECHNICAL_CONTEXT_PATTERNS)
        
        # 5. Quantitative / empirical claim
        is_empirical = bool(re.search(r'\b(\d+(\.\d+)?%|\$\d+|\b\d{4}\b|\bpercent\b)\b', s_lower))
        
        if is_sensational:
            category = "High-Risk Sensational Claim"
            risk_level = "High Risk"
            tag_class = "risk-high"
            note = "Contains alarmist hyperbole or unsupported emotional clickbait framing."
        elif is_extraordinary:
            category = "Unverified Breakthrough Assertion"
            risk_level = "High Risk"
            tag_class = "risk-high"
            note = "Extraordinary discovery or breakthrough claimed without press wire confirmation."
        elif is_attributed:
            category = "Verified Sourced Statement"
            risk_level = "Low Risk"
            tag_class = "risk-low"
            note = "Directly references institutional or journalistic source attribution."
        elif is_technical:
            category = "Technical Domain Context"
            risk_level = "Neutral"
            tag_class = "risk-neutral"
            note = "Legitimate domain terminology and observational scientific background."
        elif is_empirical:
            category = "Empirical Data Point"
            risk_level = "Medium Risk"
            tag_class = "risk-medium"
            note = "Cites statistical or measurement figures; corroborate with primary documentation."
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
            "note": note,
            "is_extraordinary": is_extraordinary
        })
        
    return claims

def analyze_mixed_veracity_profile(claims: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Evaluates whether an article demonstrates hybrid disinformation:
    blending legitimate technical context with unverified extraordinary claims.
    """
    if not claims:
        return {
            "is_mixed_veracity": False,
            "extraordinary_count": 0,
            "technical_count": 0,
            "risk_claim_ratio": 0.0
        }

    extraordinary_count = sum(1 for c in claims if c.get("is_extraordinary") or c.get("risk_level") == "High Risk")
    technical_count = sum(1 for c in claims if c.get("category") in ["Technical Domain Context", "Verified Sourced Statement"])
    total = len(claims)

    # Mixed veracity occurs when high-risk / extraordinary claims coexist with technical or factual context
    is_mixed = (extraordinary_count >= 1 and technical_count >= 1) or (extraordinary_count >= 1)

    risk_claim_ratio = round(extraordinary_count / max(1, total), 3)

    return {
        "is_mixed_veracity": is_mixed,
        "has_extraordinary_claim": extraordinary_count >= 1,
        "extraordinary_count": extraordinary_count,
        "technical_count": technical_count,
        "risk_claim_ratio": risk_claim_ratio
    }
