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

def clean_query_keywords(query: str, max_tokens: int = 4) -> str:
    """
    Extracts high-information claim entities and nouns for news search.
    Filters out conversational stopwords and sensational clickbait noise.
    """
    if not query:
        return ""
    # Remove punctuation
    cleaned = re.sub(r'[^\w\s]', ' ', query)
    tokens = [w.strip() for w in cleaned.split() if w.strip()]
    
    # Filter out stopwords, sensational buzzwords, and short words
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

def calculate_headline_similarity(query: str, headline: str) -> float:
    """
    Computes a hybrid lexical overlap and entity match score between query and headline.
    """
    q_tokens = set(re.findall(r'\b[a-zA-Z0-9]{3,}\b', query.lower())) - STOPWORDS - SENSATIONAL_NOISE
    h_tokens = set(re.findall(r'\b[a-zA-Z0-9]{3,}\b', headline.lower())) - STOPWORDS - SENSATIONAL_NOISE
    
    if not q_tokens or not h_tokens:
        return 0.0
        
    intersection = q_tokens.intersection(h_tokens)
    # Jaccard index
    jaccard = len(intersection) / len(q_tokens.union(h_tokens))
    # Recall relative to query (how much of the user's claim is represented in the headline)
    query_coverage = len(intersection) / len(q_tokens)
    
    # Weighted combination prioritizing claim coverage
    score = (0.65 * query_coverage) + (0.35 * jaccard)
    return float(round(score, 3))

def fetch_live_news_corroboration(query: str, max_results: int = 4, timeout: float = 2.0) -> Dict[str, Any]:
    """
    Queries open Google News RSS search to find real-time news reports matching the claim.
    Returns structured matches, wire authority badges, and composite corroboration score.
    """
    if not query or len(query.strip()) < 5:
        return {
            "total_matches": 0,
            "news_corroboration_score": 0.0,
            "has_wire_corroboration": False,
            "top_publishers": [],
            "articles": []
        }
        
    search_q = clean_query_keywords(query, max_tokens=4)
    if not search_q:
        search_q = query[:40]
        
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36 VeritasAI/2.0'
    }
    
    articles: List[Dict[str, Any]] = []
    top_publishers: List[str] = []
    wire_count = 0
    max_sim = 0.0

    try:
        url = f"https://news.google.com/rss/search?q={urllib.parse.quote(search_q)}&hl=en-US&gl=US&ceid=US:en"
        resp = requests.get(url, headers=headers, timeout=timeout)
        if resp.status_code == 200:
            root = ET.fromstring(resp.content)
            ch = root.find('channel')
            items = ch.findall('item') if ch is not None else []
        else:
            items = []
            
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
                wire_count += 1
            if source_name not in top_publishers:
                top_publishers.append(source_name)
                
            sim_score = calculate_headline_similarity(query, title)
            max_sim = max(max_sim, sim_score)
            
            if sim_score >= 0.40:
                match_level = "High Overlap"
            elif sim_score >= 0.20:
                match_level = "Moderate Corroboration"
            else:
                match_level = "Contextual Related"
                
            articles.append({
                "title": title,
                "source": source_name,
                "pub_date": pub_date,
                "link": link,
                "match_score": sim_score,
                "match_level": match_level,
                "is_wire_source": is_wire
            })
    except Exception:
        # Resilient fallback: return empty structure without crashing
        pass
        
    # Corroboration score calculation (0.0 to 1.0)
    # Factors: match similarity, number of articles, presence of major wire/press
    if not articles:
        corroboration_score = 0.0
    else:
        base_score = max_sim * 0.70
        volume_bonus = min(len(articles) * 0.05, 0.15)
        wire_bonus = 0.15 if wire_count > 0 else 0.0
        corroboration_score = round(min(base_score + volume_bonus + wire_bonus, 1.0), 3)
        
    return {
        "total_matches": len(articles),
        "news_corroboration_score": corroboration_score,
        "has_wire_corroboration": (wire_count > 0),
        "top_publishers": top_publishers[:4],
        "articles": articles
    }
