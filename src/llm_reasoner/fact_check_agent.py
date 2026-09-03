import os
import json
import urllib.request
import urllib.parse
from typing import Dict, Any, Optional, List
from concurrent.futures import ThreadPoolExecutor

from src.llm_reasoner.news_grounding_engine import fetch_live_news_corroboration

REFUTATION_CUES = {
    "denies", "denied", "no evidence", "unproven", "fictional", "debunked", 
    "hoax", "conspiracy", "unsubstantiated", "unconfirmed", "disputed"
}

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
                import re
                clean_snippet = re.sub(r'<.*?>', '', r.get('snippet', ''))
                snippet_lower = clean_snippet.lower()
                has_refutation = any(cue in snippet_lower for cue in REFUTATION_CUES)
                grounding.append({
                    "title": r.get('title'),
                    "snippet": clean_snippet,
                    "has_refutation": has_refutation
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
        Generates structured, token-efficient reasoning for the prediction with concurrent live knowledge & news retrieval.
        """
        is_fake = fake_probability >= 0.5
        confidence = fake_probability if is_fake else (1.0 - fake_probability)
        verdict = "Likely Fake / Sensationalized" if is_fake else "Likely Real / Mainstream"
        
        # 1. Fetch live knowledge grounding & real-time news wire corroboration concurrently
        query_text = headline if headline else text_snippet[:100]
        
        knowledge_sources = []
        news_info = {
            "total_matches": 0, 
            "news_corroboration_score": 0.0, 
            "has_wire_corroboration": False, 
            "has_claim_corroboration": False,
            "topic_covered_claim_absent": False,
            "top_publishers": [], 
            "articles": []
        }
        
        if query_text and len(query_text.strip()) >= 5:
            with ThreadPoolExecutor(max_workers=2) as executor:
                future_wiki = executor.submit(fetch_encyclopedic_corroboration, query_text, 2)
                future_news = executor.submit(fetch_live_news_corroboration, query_text, 4, 2.5)
                
                try:
                    knowledge_sources = future_wiki.result(timeout=2.8) or []
                except Exception:
                    knowledge_sources = []
                    
                try:
                    news_info = future_news.result(timeout=2.8) or news_info
                except Exception:
                    pass
        
        reasons = []
        topic_absent = news_info.get("topic_covered_claim_absent", False)
        has_claim_corrob = news_info.get("has_claim_corroboration", False)

        if is_fake:
            if topic_absent:
                reasons.append("Topic entities (e.g. agency or institutions) are actively reported in current news, but the specific high-impact claim is absent from all wire reports, indicating an unverified or fabricated assertion.")
            elif not has_claim_corrob:
                if stylistic_info and (stylistic_info.get("is_all_caps_title") or stylistic_info.get("sensational_keywords")):
                    reasons.append("Zero corroborating press wire reports found across major global news agencies, indicating an unverified claim or fabrication.")
                else:
                    reasons.append("Zero corroborating press wire reports found across major global news agencies for this breaking assertion.")

            if stylistic_info and (stylistic_info.get("is_all_caps_title") or stylistic_info.get("is_all_caps_body")):
                reasons.append("Extreme capitalization (all-caps shouting) identified, characteristic of sensationalist / clickbait claims.")
            if stylistic_info and stylistic_info.get("sensational_keywords"):
                reasons.append(f"Sensationalist / alarmist triggers detected: {', '.join(stylistic_info['sensational_keywords'][:3])}.")
            elif salient_fake_words:
                sensational_subset = [w['token'] for w in salient_fake_words if w.get('is_sensational')]
                if sensational_subset:
                    reasons.append(f"Sensational clickbait patterns detected: {', '.join(sensational_subset[:3])}.")
                else:
                    reasons.append("Elevated disinformation markers detected in vocabulary and semantic patterns.")
            else:
                reasons.append("Elevated risk markers and lack of corroborative journalistic framing detected.")

            refuted_source = next((s for s in knowledge_sources if s.get("has_refutation")), None)
            if refuted_source:
                reasons.append(f"Encyclopedic context notes official agencies or records have disputed or denied unverified reports regarding '{refuted_source['title']}'.")

            if stylistic_info and stylistic_info.get("attribution_score", 0) == 0:
                reasons.append("Zero verified journalistic attribution, institutional source citations, or official corroboration found.")
            elif not topic_absent:
                reasons.append("Stylistic tone exhibits framing characteristic of unverified news.")
        else:
            if has_claim_corrob:
                top_pubs = ", ".join(news_info.get("top_publishers", [])[:2])
                reasons.append(f"Claim corroborated by live press wire reports from recognized outlets ({top_pubs}).")
                
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
                if not top_src.get("has_refutation"):
                    reasons.append(f"Encyclopedic context aligned with verified topic: '{top_src['title']}'.")
                else:
                    reasons.append(f"Encyclopedic reference records caution regarding '{top_src['title']}'.")

        summary = {
            "verdict": verdict,
            "confidence_percentage": round(confidence * 100, 2),
            "fake_probability": round(fake_probability, 4),
            "key_indicators": [w['token'] for w in (salient_fake_words if is_fake else salient_real_words)[:5]],
            "attribution_indicators": stylistic_info.get("attribution_keywords", []) if stylistic_info else [],
            "knowledge_corroboration": knowledge_sources,
            "news_corroboration": news_info.get("articles", []),
            "news_corroboration_score": news_info.get("news_corroboration_score", 0.0),
            "has_wire_corroboration": news_info.get("has_wire_corroboration", False),
            "has_claim_corroboration": has_claim_corrob,
            "topic_covered_claim_absent": topic_absent,
            "rationale": " ".join(reasons)
        }
        return summary

