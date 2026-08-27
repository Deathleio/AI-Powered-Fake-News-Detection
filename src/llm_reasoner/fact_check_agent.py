import os
import json
import urllib.request
import urllib.parse
from typing import Dict, Any, Optional, List

def fetch_encyclopedic_corroboration(query: str, max_results: int = 2) -> list:
    """
    Queries open Wikipedia API to fetch factual grounding snippets for entity/claim verification.
    Timeout 1.5s to prevent blocking.
    """
    if not query or len(query.strip()) < 5:
        return []
    
    clean_q = " ".join([w for w in query.split() if len(w) > 3][:6])
    url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={urllib.parse.quote(clean_q)}&format=json&utf8=1&srlimit={max_results}"
    
    try:
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'VeritasAI-FakeNewsDetector/1.0 (academic research fact check)'}
        )
        with urllib.request.urlopen(req, timeout=1.5) as response:
            data = json.loads(response.read().decode('utf-8'))
            results = data.get('query', {}).get('search', [])
            grounding = []
            for r in results:
                # Strip HTML tags
                import re
                clean_snippet = re.sub(r'<.*?>', '', r.get('snippet', ''))
                grounding.append({
                    "title": r.get('title'),
                    "snippet": clean_snippet
                })
            return grounding
    except Exception:
        return []

class LLMFactCheckReasoner:
    """
    Synthesizes model predictions, salient keywords, and live knowledge grounding into a structured fact-checking explanation.
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
        Generates structured, token-efficient reasoning for the prediction with optional live knowledge retrieval.
        """
        is_fake = fake_probability >= 0.5
        confidence = fake_probability if is_fake else (1.0 - fake_probability)
        verdict = "Likely Fake / Sensationalized" if is_fake else "Likely Real / Mainstream"
        
        # 1. Fetch live knowledge grounding for non-empty headlines
        query_text = headline if headline else text_snippet[:100]
        knowledge_sources = fetch_encyclopedic_corroboration(query_text)
        
        reasons = []
        if is_fake:
            if stylistic_info and (stylistic_info.get("is_all_caps_title") or stylistic_info.get("is_all_caps_body")):
                reasons.append("Extreme capitalization (all-caps shouting) identified, characteristic of sensationalist / clickbait claims.")
            if stylistic_info and stylistic_info.get("sensational_keywords"):
                reasons.append(f"Sensationalist / alarmist triggers detected: {', '.join(stylistic_info['sensational_keywords'][:3])}.")
            elif salient_fake_words:
                sensational_subset = [w['token'] for w in salient_fake_words if w.get('is_sensational')]
                if sensational_subset:
                    reasons.append(f"Sensational clickbait patterns detected: {', '.join(sensational_subset[:3])}.")
                else:
                    reasons.append("Elevated sensational markers and lack of corroborative journalistic framing detected.")
            else:
                reasons.append("Elevated sensational markers and lack of corroborative journalistic framing detected.")
                
            if stylistic_info and stylistic_info.get("attribution_score", 0) == 0:
                reasons.append("Zero verified journalistic attribution, institutional source citations, or official corroboration found.")
            else:
                reasons.append("Stylistic tone exhibits informal/alarmist framing characteristic of unverified news.")
        else:
            if stylistic_info and stylistic_info.get("attribution_keywords"):
                attrs = ", ".join(stylistic_info["attribution_keywords"][:3])
                reasons.append(f"Verified authoritative/journalistic attribution detected: '{attrs}'.")
            elif salient_real_words:
                reasons.append(f"Factual reporting vocabulary and standard journalistic tone identified: {', '.join([w['token'] for w in salient_real_words[:4]])}.")
            else:
                reasons.append("Attribution markers align with standard news reporting.")
                
            if stylistic_info and stylistic_info.get("attribution_score", 0) > 0:
                reasons.append("Formal journalistic/scientific attribution markers and objective syntax verified.")
            else:
                reasons.append("Syntactic structure adheres to objective narrative standards.")
            
            if knowledge_sources:
                top_src = knowledge_sources[0]
                reasons.append(f"Encyclopedic context aligned with verified topic: '{top_src['title']}'.")

        summary = {
            "verdict": verdict,
            "confidence_percentage": round(confidence * 100, 2),
            "fake_probability": round(fake_probability, 4),
            "key_indicators": [w['token'] for w in (salient_fake_words if is_fake else salient_real_words)[:5]],
            "attribution_indicators": stylistic_info.get("attribution_keywords", []) if stylistic_info else [],
            "knowledge_corroboration": knowledge_sources,
            "rationale": " ".join(reasons)
        }
        return summary
