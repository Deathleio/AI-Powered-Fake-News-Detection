"""
Test suite for Live News Grounding Engine and Multi-Engine Fact Checking Agent.
"""
import time
import pytest
from src.llm_reasoner.news_grounding_engine import fetch_live_news_corroboration, clean_query_keywords, calculate_headline_similarity
from src.llm_reasoner.fact_check_agent import LLMFactCheckReasoner, fetch_encyclopedic_corroboration

def test_clean_query_keywords():
    q = "The Federal Reserve holds benchmark interest rates steady for now!"
    cleaned = clean_query_keywords(q)
    assert "Federal" in cleaned
    assert "Reserve" in cleaned
    assert "!" not in cleaned

def test_calculate_headline_similarity():
    q = "Federal Reserve holds interest rates steady"
    h1 = "Federal Reserve maintains interest rates steady in unanimous vote"
    h2 = "Celebrity chef opens luxury restaurant in downtown Paris"
    
    sim1 = calculate_headline_similarity(q, h1)
    sim2 = calculate_headline_similarity(q, h2)
    assert sim1 > 0.3
    assert sim2 == 0.0

def test_fetch_live_news_corroboration_real_claim():
    claim = "NASA James Webb Space Telescope water vapor"
    res = fetch_live_news_corroboration(claim, max_results=3, timeout=6.0)
    assert isinstance(res, dict)
    assert "news_corroboration_score" in res
    assert "articles" in res
    assert res["total_matches"] > 0
    assert res["news_corroboration_score"] > 0.1

def test_fetch_live_news_corroboration_fake_claim():
    fake_claim = "Secret Ancient Root Cures All Disease Overnight Miracle 100%"
    res = fetch_live_news_corroboration(fake_claim, max_results=3, timeout=6.0)
    assert res["has_wire_corroboration"] is False
    assert res["news_corroboration_score"] < 0.35
    # Confirm none of the articles had high overlap
    assert not any(a["match_level"] == "High Overlap" for a in res["articles"])

def test_concurrent_reasoner_execution_timing():
    reasoner = LLMFactCheckReasoner()
    start = time.time()
    verdict = reasoner.synthesize_verdict(
        headline="Federal Reserve holds interest rates steady",
        text_snippet="The central bank voted to keep rates unchanged.",
        fake_probability=0.10,
        salient_fake_words=[],
        salient_real_words=[{"token": "federal"}, {"token": "rates"}],
        stylistic_info={"attribution_score": 0.8}
    )
    elapsed = time.time() - start
    # Concurrent execution should complete well under 4 seconds
    assert elapsed < 4.0
    assert "news_corroboration" in verdict
    assert "knowledge_corroboration" in verdict
    assert verdict["verdict"] == "Likely Real / Mainstream"
    assert isinstance(verdict["news_corroboration"], list)

if __name__ == "__main__":
    pytest.main(["-v", __file__])
