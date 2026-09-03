"""
Test suite for Overhauled Web Link Pipeline and Article Extractor.
"""
import pytest
from src.data.url_extractor import extract_article_from_url, clean_and_normalize_url

def test_clean_and_normalize_url():
    raw = "http://www.bbc.com/news/world-123?utm_source=facebook&utm_medium=cpc&fbclid=xyz#section2"
    cleaned = clean_and_normalize_url(raw)
    assert "utm_source" not in cleaned
    assert "fbclid" not in cleaned
    assert "#section2" not in cleaned
    assert cleaned.startswith("https://") or cleaned.startswith("http://")

def test_extract_article_wikipedia():
    url = "https://en.wikipedia.org/wiki/Artificial_intelligence"
    res = extract_article_from_url(url, timeout=5.0)
    assert res["success"] is True
    assert res["domain"] == "en.wikipedia.org"
    assert "software" in res["title"].lower() or "intelligence" in res["title"].lower() or "artificial" in res["title"].lower()
    assert res["word_count"] > 200
    assert res["reading_time_min"] > 1

def test_extract_article_invalid_or_404():
    url = "https://httpstat.us/404"
    res = extract_article_from_url(url, timeout=4.0)
    assert res["success"] is False
    assert res["error"] is not None

def test_extract_article_empty_url():
    res = extract_article_from_url("")
    assert res["success"] is False

if __name__ == "__main__":
    pytest.main(["-v", __file__])
