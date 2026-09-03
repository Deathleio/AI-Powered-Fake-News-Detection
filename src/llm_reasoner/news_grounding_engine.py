"""
Real-time Live News Grounding & Cross-Corroboration Engine.
Queries open-source news feed APIs (Google News Open Search Feed) to corroborate
breaking claims, wire reporting, and current events without requiring API keys.
"""

import re
import html
import urllib.parse
import xml.etree.ElementTree as ET
from typing import Dict, Any, List, Optional
import requests

from src.credibility.domain_registry import DOMAIN_DATABASE

# Major verified wire services and premier publishers
WIRE_SERVICES = {
    "reuters", "associated press", "ap", "afp", "agence france-presse",
    "bloomberg", "bbc", "bbc news", "the wall street journal", "wsj",
    "the new york times", "the washington post", "the guardian", "npr",
    "pbs", "cbs news", "abc news", "nbc news", "cnn", "the associated press",
    "financial times", "nature", "science", "the lancet", "who", "cdc", "nasa"
}

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", "with",
    "by", "about", "against", "between", "into", "through", "during", "before",
    "after", "above", "below", "from", "up", "down", "of", "off", "over", "under",
    "again", "further", "then", "once", "here", "there", "when", "where", "why",
    "how", "all", "any", "both", "each", "few", "more", "most", "other", "some",
    "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very",
    "s", "t", "can", "will", "just", "don", "should", "now", "says", "said", "new",
    "report", "reports", "state", "states", "year", "years"
}

SENSATIONAL_NOISE = {
    "miracle", "shocking", "unbelievable", "secret", "must see", "truth",
    "revealed", "banned", "suppressed", "mindblowing", "wont believe", "exposed"
}

def clean_query_keywords(query: str, max_tokens: int = 8) -> str:
    """
    Extracts high-information claim entities and predicates for news search.
    Preserves core claim assertions while filtering out conversational stopwords.
    """
    if not query:
        return ""
    # Remove punctuation
    cleaned = re.sub(r'[^\w\s]', ' ', query)
    tokens = [w.strip() for w in cleaned.split() if w.strip()]
    
    # Filter out stopwords, sensational noise, and very short tokens
    informative = [
        w for w in tokens 
        if w.lower() not in STOPWORDS 
        and w.lower() not in SENSATIONAL_NOISE
        and len(w) > 2
    ]
    if not informative:
        informative = [w for w in tokens if w.lower() not in STOPWORDS and len(w) > 2]
    if not informative:
        informative = tokens[:max_tokens]
    return " ".join(informative[:max_tokens])

# Common institutional or topical background entities that frequently appear in news
COMMON_ENTITIES = {
    "nasa", "mars", "rover", "moon", "space", "biden", "trump", "senate", "congress",
    "white", "house", "pentagon", "fed", "federal", "reserve", "bank", "police",
    "government", "ukraine", "russia", "china", "who", "cdc", "fbi", "court",
    "openai", "google", "microsoft", "apple", "anthropic", "meta", "nvidia", "intel", "tesla"
}

class HeadlineMatchResult(float):
    """
    Dual-type match result that behaves as a float (backward compatible)
    and unpacks as a 3-tuple: (score, match_level, claim_matched).
    """
    match_level: str
    claim_matched: bool

    def __new__(cls, score: float, match_level: str = "Contextual Related", claim_matched: bool = False):
        instance = super().__new__(cls, score)
        instance.match_level = match_level
        instance.claim_matched = claim_matched
        return instance

    def __iter__(self):
        yield float(self)
        yield self.match_level
        yield self.claim_matched

def calculate_headline_similarity(query: str, headline: str) -> HeadlineMatchResult:
    """
    Computes a hybrid lexical overlap and entity match score between query and headline.
    Distinguishes between broad topic alignment (matching background entities) and
    actual claim corroboration (matching the specific breakthrough or assertion predicates).

    Returns:
        HeadlineMatchResult (float with match_level and claim_matched attributes, iterable as 3-tuple)
    """
    q_tokens = set(re.findall(r'\b[a-zA-Z0-9]{3,}\b', query.lower())) - STOPWORDS - SENSATIONAL_NOISE
    h_tokens = set(re.findall(r'\b[a-zA-Z0-9]{3,}\b', headline.lower())) - STOPWORDS - SENSATIONAL_NOISE
    
    if not q_tokens or not h_tokens:
        return HeadlineMatchResult(0.0, "No Match", False)
        
    intersection = q_tokens.intersection(h_tokens)
    if not intersection:
        return HeadlineMatchResult(0.0, "No Match", False)

    # Separate background entities from core claim assertion tokens
    claim_tokens = {t for t in q_tokens if t not in COMMON_ENTITIES}
    claim_intersection = claim_tokens.intersection(h_tokens) if claim_tokens else set()

    # Jaccard index
    jaccard = len(intersection) / len(q_tokens.union(h_tokens))
    # Recall relative to query
    query_coverage = len(intersection) / len(q_tokens)
    
    # Claim coverage: how many of the non-background claim predicates are in the headline
    claim_coverage = len(claim_intersection) / len(claim_tokens) if claim_tokens else query_coverage

    # If the user made a specific claim (e.g. "fossilised biological structures", "cures all", "ban cash")
    # but the headline matches ONLY background entities (e.g. "NASA", "Rover", "Mars"),
    # this is topic coverage, NOT claim corroboration!
    claim_matched = False
    if claim_tokens:
        if len(claim_intersection) >= 2 or (len(claim_tokens) == 1 and len(claim_intersection) == 1):
            claim_matched = True
        elif claim_coverage >= 0.40:
            claim_matched = True

    if claim_matched:
        score = (0.50 * claim_coverage) + (0.30 * query_coverage) + (0.20 * jaccard)
        score = float(round(score, 3))
        if score >= 0.40:
            match_level = "High Overlap"
        elif score >= 0.20:
            match_level = "Moderate Corroboration"
        else:
            match_level = "Contextual Related"
    else:
        # Topic matched but the core claim assertion was absent
        score = float(round(min(0.18, 0.25 * query_coverage), 3))
        match_level = "Topic Only (Claim Absent)"

    return HeadlineMatchResult(score, match_level, claim_matched)

def fetch_live_news_corroboration(query: str, max_results: int = 4, timeout: float = 2.5) -> Dict[str, Any]:
    """
    Queries open Google News RSS search to find real-time news reports matching the claim.
    Returns structured matches, wire authority badges, and composite corroboration score.
    """
    if not query or len(query.strip()) < 5:
        return {
            "total_matches": 0,
            "news_corroboration_score": 0.0,
            "has_wire_corroboration": False,
            "has_claim_corroboration": False,
            "topic_covered_claim_absent": False,
            "top_publishers": [],
            "articles": []
        }
        
    search_q = clean_query_keywords(query, max_tokens=6)
    if not search_q:
        search_q = query[:50]
        
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36 VeritasAI/2.0'
    }
    
    articles: List[Dict[str, Any]] = []
    top_publishers: List[str] = []
    wire_corroborating_count = 0
    total_wire_count = 0
    max_claim_sim = 0.0
    any_claim_corroborated = False

    try:
        # Build candidate queries: full clean query and structured entity fallbacks
        tokens = search_q.split()
        candidate_queries = [search_q]
        if len(tokens) > 3:
            candidate_queries.append(" ".join(tokens[:4]))
            if tokens[0].lower() in ["nasa", "us", "the", "new"] and len(tokens) > 4:
                candidate_queries.append(" ".join(tokens[1:5]))
            elif tokens[0].lower() in ["nasa", "us", "the", "new"]:
                candidate_queries.append(" ".join(tokens[1:4]))

        items = []
        for q_try in candidate_queries:
            url = f"https://news.google.com/rss/search?q={urllib.parse.quote(q_try)}&hl=en-US&gl=US&ceid=US:en"
            resp = requests.get(url, headers=headers, timeout=timeout)
            if resp.status_code == 200:
                root = ET.fromstring(resp.content)
                ch = root.find('channel')
                items = ch.findall('item') if ch is not None else []
                if items:
                    break
            
        for item in items[:max_results]:
            raw_title = item.findtext('title') or ""
            # Strip publisher suffix (e.g. "Headline - Reuters")
            title = re.sub(r'\s*[-|–—]\s*[^-|–—]+$', '', raw_title).strip()
            if not title:
                title = raw_title
                
            source_el = item.find('source')
            source_name = source_el.text.strip() if (source_el is not None and source_el.text) else "News Source"
            pub_date = item.findtext('pubDate') or ""
            link = item.findtext('link') or ""
            
            # Check if wire service or authoritative outlet
            is_wire = any(w in source_name.lower() for w in WIRE_SERVICES)
            if is_wire:
                total_wire_count += 1
            if source_name not in top_publishers:
                top_publishers.append(source_name)
                
            sim_score, match_level, claim_matched = calculate_headline_similarity(query, title)
            if claim_matched:
                any_claim_corroborated = True
                max_claim_sim = max(max_claim_sim, sim_score)
                if is_wire:
                    wire_corroborating_count += 1
                
            articles.append({
                "title": title,
                "source": source_name,
                "pub_date": pub_date,
                "link": link,
                "match_score": sim_score,
                "match_level": match_level,
                "claim_matched": claim_matched,
                "is_wire_source": is_wire
            })
    except Exception:
        # Resilient fallback: return empty structure without crashing
        pass
        
    # Corroboration score calculation (0.0 to 1.0)
    # ONLY rewarded if the claim itself is corroborated, not just background topic
    if not articles or not any_claim_corroborated:
        corroboration_score = 0.0
    else:
        base_score = max_claim_sim * 0.70
        volume_bonus = min(sum(1 for a in articles if a.get("claim_matched")) * 0.08, 0.16)
        wire_bonus = 0.15 if wire_corroborating_count > 0 else 0.0
        corroboration_score = round(min(base_score + volume_bonus + wire_bonus, 1.0), 3)

    # Flag: topic was found in news / wire coverage, but 0 articles corroborated the specific claim
    topic_covered_claim_absent = (len(articles) > 0 and not any_claim_corroborated)
        
    return {
        "total_matches": len(articles),
        "news_corroboration_score": corroboration_score,
        "has_wire_corroboration": (wire_corroborating_count > 0),
        "has_claim_corroboration": any_claim_corroborated,
        "topic_covered_claim_absent": topic_covered_claim_absent,
        "top_publishers": top_publishers[:4],
        "articles": articles
    }
