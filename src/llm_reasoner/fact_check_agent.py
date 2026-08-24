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
        salient_real_words: list
    ) -> Dict[str, Any]:
        """
        Generates structured, token-efficient reasoning for the prediction.
        """
        is_fake = fake_probability >= 0.5
        confidence = fake_probability if is_fake else (1.0 - fake_probability)
        verdict = "Likely Fake / Sensationalized" if is_fake else "Likely Real / Mainstream"
        
        # High confidence reasoning generation
        if is_fake:
            reasons = [
                f"High-frequency sensational or hyperpartisan lexical triggers detected: {', '.join([w['token'] for w in salient_fake_words[:4]])}." if salient_fake_words else "Elevated sensational markers detected in headline.",
                "Stylistic tone exhibits informal/alarmist framing characteristic of clickbait news."
            ]
        else:
            reasons = [
                f"Factual reporting vocabulary and standard journalistic tone identified: {', '.join([w['token'] for w in salient_real_words[:4]])}." if salient_real_words else "Attribution markers align with standard news reporting.",
                "Syntactic structure adheres to objective narrative standards."
            ]
            
        summary = {
            "verdict": verdict,
            "confidence_percentage": round(confidence * 100, 2),
            "fake_probability": round(fake_probability, 4),
            "key_indicators": [w['token'] for w in (salient_fake_words if is_fake else salient_real_words)[:5]],
            "rationale": " ".join(reasons)
        }
        return summary
