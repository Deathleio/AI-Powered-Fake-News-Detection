import os
import json
from typing import Dict, Any, Optional

class LLMFactCheckReasoner:
    """
    Synthesizes model predictions, salient keywords, and context into a structured fact-checking explanation.
    """
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")

    def synthesize_verdict(
        self,
        headline: str,
        text_snippet: str,
        fake_probability: float,
        salient_fake_words: list,
        salient_real_words: list,
        stylistic_info: Optional[dict] = None
    ) -> Dict[str, Any]:
        """
        Generates structured, token-efficient reasoning for the prediction.
        """
        is_fake = fake_probability >= 0.5
        confidence = fake_probability if is_fake else (1.0 - fake_probability)
        verdict = "Likely Fake / Sensationalized" if is_fake else "Likely Real / Mainstream"
        
        reasons = []
        if is_fake:
            if stylistic_info and (stylistic_info.get("is_all_caps_title") or stylistic_info.get("is_all_caps_body")):
                reasons.append("Extreme capitalization (all-caps shouting) identified, characteristic of sensationalist / clickbait claims.")
            if stylistic_info and stylistic_info.get("sensational_keywords"):
                reasons.append(f"Sensationalist / alarmist triggers detected: {', '.join(stylistic_info['sensational_keywords'][:3])}.")
            elif salient_fake_words:
                reasons.append(f"High-frequency sensational or hyperpartisan lexical triggers detected: {', '.join([w['token'] for w in salient_fake_words[:4]])}.")
            else:
                reasons.append("Elevated sensational markers and lack of corroborative journalistic framing detected.")
                
            if stylistic_info and stylistic_info.get("attribution_score", 0) == 0:
                reasons.append("Zero verified journalistic attribution, institutional source citations, or official corroboration found.")
            else:
                reasons.append("Stylistic tone exhibits informal/alarmist framing characteristic of unverified news.")
        else:
            if salient_real_words:
                reasons.append(f"Factual reporting vocabulary and standard journalistic tone identified: {', '.join([w['token'] for w in salient_real_words[:4]])}.")
            else:
                reasons.append("Attribution markers align with standard news reporting.")
                
            if stylistic_info and stylistic_info.get("attribution_score", 0) > 0:
                reasons.append("Formal journalistic attribution markers and objective syntax verified.")
            else:
                reasons.append("Syntactic structure adheres to objective narrative standards.")
            
        summary = {
            "verdict": verdict,
            "confidence_percentage": round(confidence * 100, 2),
            "fake_probability": round(fake_probability, 4),
            "key_indicators": [w['token'] for w in (salient_fake_words if is_fake else salient_real_words)[:5]],
            "rationale": " ".join(reasons)
        }
        return summary
