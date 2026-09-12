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
    "report", "reports", "state", "states", "year", "years", "pop", "star", "icon",
    "announces", "announced", "announcing", "announcement", "reveals", "revealed", "revealing",
    "confirms", "confirmed", "confirming", "claims", "claimed", "claiming",
    "surprise", "project", "major", "latest", "first", "look", "drops",
    "launch", "launches", "launched", "launching",
    "unveil", "unveils", "unveiled", "unveiling",
    "introduce", "introduces", "introduced", "introducing",
    "release", "releases", "released", "releasing",
    "that", "this", "these", "those", "it", "its", "they", "them", "their",
    "official", "update", "updates", "video", "photos", "watch", "see", "fans", "insiders"
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

# Common institutional, geographical, and high-profile public entities
COMMON_ENTITIES = {
    # Government & International
    "nasa", "mars", "rover", "moon", "space", "biden", "trump", "senate", "congress",
    "white", "house", "pentagon", "fed", "federal", "reserve", "bank", "police",
    "government", "ukraine", "russia", "china", "who", "cdc", "fbi", "court",
    "sec", "ftc", "faa", "un", "nato", "eu", "imf",
    # Tech & AI Giants
    "crowdstrike", "openai", "google", "microsoft", "apple", "anthropic", "meta", 
    "nvidia", "intel", "amd", "tesla", "amazon", "cisco", "oracle", "ibm",
    "salesforce", "adobe", "palantir", "cloudflare", "uber", "spacex", "deepmind",
    "samsung", "huawei", "bytedance", "tiktok", "twitter", "x", "tsmc", "broadcom",
    # Industry & Pharma
    "boeing", "lockheed", "pfizer", "moderna", "astrazeneca", "johnson", "bayer",
    # Prominent Figures
    "rihanna", "swift", "taylor", "musk", "elon", "bezos", "gates", "zuckerberg", "altman", "huang"
}

# Generic domain topic nouns and context words that do not by themselves constitute a unique claim assertion
DOMAIN_TOPIC_NOUNS = {
    "cybersecurity", "security", "vulnerability", "vulnerabilities", "software", "hardware",
    "operating", "system", "systems", "enterprise", "technology", "technologies", "cloud",
    "network", "networks", "data", "digital", "service", "services", "platform", "platforms",
    "company", "firm", "corp", "corporation", "inc", "ltd", "ceo", "cto", "executive", "executives",
    "industry", "market", "markets", "stock", "stocks", "shares", "investors", "economy",
    "economic", "policy", "policies", "program", "programs", "department", "agency", "agencies",
    "official", "officials", "spokesperson", "spokesman", "statement", "announcement",
    "research", "researchers", "study", "studies", "scientists", "experts", "analysts",
    "device", "devices", "computer", "computers", "internet", "web", "online", "app", "apps",
    "application", "applications", "tool", "tools", "product", "products", "project", "projects",
    "model", "models", "algorithm", "algorithms", "ai", "artificial", "intelligence", "telescope",
    "telescopes", "satellite", "satellites", "rates", "rate", "interest"
}

def _stem_word(w: str) -> str:
    """Lightweight suffix normalizer for common English inflections."""
    w = w.lower()
    for suff in ("ies", "es", "s", "ed", "ing"):
        if w.endswith(suff) and len(w) - len(suff) >= 3:
            return w[:-len(suff)]
    return w

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

NATION_ENTITIES = {
    "switzerland", "swiss", "syria", "syrian", "uk", "britain", "british", "england", "english",
    "us", "usa", "america", "american", "china", "chinese", "russia", "russian",
    "japan", "japanese", "germany", "german", "france", "french", "india", "indian",
    "canada", "canadian", "australia", "australian", "israel", "israeli", "iran", "iranian",
    "turkey", "turkish", "brazil", "brazilian", "mexico", "mexican", "italy", "italian"
}

def calculate_headline_similarity(query: str, headline: str) -> HeadlineMatchResult:
    """
    Computes a strict lexical overlap and entity match score between query and headline.
    Strictly distinguishes between broad topic alignment (matching entities or generic domain nouns)
    and actual claim corroboration (matching the specific breakthrough or assertion predicates).

    Returns:
        HeadlineMatchResult (float with match_level and claim_matched attributes, iterable as 3-tuple)
    """
    q_tokens = set(re.findall(r'\b[a-zA-Z0-9]{3,}\b', query.lower())) - STOPWORDS - SENSATIONAL_NOISE
    h_tokens = set(re.findall(r'\b[a-zA-Z0-9]{3,}\b', headline.lower())) - STOPWORDS - SENSATIONAL_NOISE
    
    if not q_tokens or not h_tokens:
        return HeadlineMatchResult(0.0, "No Match", False)
        
    q_stems = {_stem_word(t) for t in q_tokens}
    h_stems = {_stem_word(t) for t in h_tokens}
    stem_intersection = q_stems.intersection(h_stems)
    if not stem_intersection:
        return HeadlineMatchResult(0.0, "No Match", False)

    # Disallow cross-national entity confusion (e.g. Syrian central bank cannot corroborate Swiss central bank)
    q_nations = q_tokens.intersection(NATION_ENTITIES)
    h_nations = h_tokens.intersection(NATION_ENTITIES)
    if q_nations and h_nations and not q_nations.intersection(h_nations):
        return HeadlineMatchResult(0.0, "No Match", False)

    # Identify entity stems and domain stems
    entity_stems = {_stem_word(t) for t in q_tokens if t in COMMON_ENTITIES or t in NATION_ENTITIES}
    entity_matched = bool(entity_stems.intersection(stem_intersection))

    stop_stems = {_stem_word(s) for s in STOPWORDS}
    # Core assertion tokens: specific action predicates, findings, or claims (NOT entities, NOT generic domain nouns, NOT stopwords)
    assertion_stems = {_stem_word(t) for t in q_tokens if t not in COMMON_ENTITIES and t not in NATION_ENTITIES and t not in DOMAIN_TOPIC_NOUNS} - stop_stems
    assertion_matched = assertion_stems.intersection(stem_intersection)

    query_coverage = len(stem_intersection) / len(q_stems)
    jaccard = len(stem_intersection) / len(q_stems.union(h_stems))

    # Strict assertion corroboration rule:
    # A claim CANNOT be matched solely on background entity and generic domain tokens.
    claim_matched = False
    if assertion_stems:
        if len(assertion_stems) >= 3:
            claim_matched = (len(assertion_matched) >= 2) or (len(assertion_matched) >= 1 and query_coverage >= 0.50)
        else:
            claim_matched = len(assertion_matched) >= 1 and query_coverage >= 0.35
    else:
        # Query only contained entity and domain tokens
        claim_matched = (query_coverage >= 0.50)

    if claim_matched:
        assertion_cov = len(assertion_matched) / len(assertion_stems) if assertion_stems else query_coverage
        score = (0.50 * assertion_cov) + (0.30 * query_coverage) + (0.20 * jaccard)
        score = float(round(min(score, 1.0), 3))
        if score >= 0.40:
            match_level = "High Overlap"
        elif score >= 0.20:
            match_level = "Moderate Corroboration"
        else:
            match_level = "Contextual Related"
    else:
        # If entities or domain nouns matched but the core assertion is absent
        if entity_matched or len(stem_intersection) >= 1:
            score = float(round(min(0.18, 0.25 * query_coverage), 3))
            match_level = "Topic Only (Claim Absent)"
        else:
            score = 0.0
            match_level = "No Match"

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

    # Flag: topic entities were found in news / wire coverage, but 0 articles corroborated the specific claim
    has_topic_match = any(a.get("match_level") == "Topic Only (Claim Absent)" for a in articles)
    topic_covered_claim_absent = (has_topic_match and not any_claim_corroborated)
        
    return {
        "total_matches": len(articles),
        "news_corroboration_score": corroboration_score,
        "has_wire_corroboration": (wire_corroborating_count > 0),
        "has_claim_corroboration": any_claim_corroborated,
        "topic_covered_claim_absent": topic_covered_claim_absent,
        "top_publishers": top_publishers[:4],
        "articles": articles
    }
