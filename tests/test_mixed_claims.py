import pytest
from src.serving.api import NewsArticleRequest, explain_news, predict_news
from src.explainability.claim_segmenter import segment_and_analyze_claims, analyze_mixed_veracity_profile
from src.llm_reasoner.news_grounding_engine import calculate_headline_similarity

def test_user_partially_fake_mars_sample():
    title = "NASA Mars Rover Discovery Confirms Fossilised Biological Structures"
    text = (
        "formations buried beneath the Martian regolith, measuring several meters in height "
        "and displaying repeating patterns inconsistent with natural geological erosion. "
        "Initial spectral analysis from the SuperCam instrument suggests the presence of "
        "organic molecular clusters deeply embedded within the crystalline layers of the structures. "
        "NASA has temporarily paused the rover's planned trajectory to investigate further."
    )
    req = NewsArticleRequest(title=title, text=text)
    res = explain_news(req)

    assert res.is_fake is True
    assert res.verdict in ["Partially Fake / Misleading", "Fake News"]
    assert res.veritas_score <= 45
    assert res.fake_probability >= 0.60
    assert len(res.claims_breakdown) > 0

    categories = [c["category"] for c in res.claims_breakdown]
    assert any("Breakthrough" in cat or "High-Risk" in cat for cat in categories)

def test_claim_segmenter_mixed_profile():
    title = "NASA Mars Rover Discovery Confirms Fossilised Biological Structures"
    text = "SuperCam instrument indicates crystalline layers in the Martian regolith."
    claims = segment_and_analyze_claims(title, text)
    profile = analyze_mixed_veracity_profile(claims)

    assert profile["is_mixed_veracity"] is True
    assert profile["extraordinary_count"] >= 1
    assert profile["technical_count"] >= 1

def test_headline_similarity_claim_vs_topic():
    query = "NASA Mars Rover Discovery Confirms Fossilised Biological Structures"
    
    generic_topic_headline = "NASA's Curiosity Rover Captures Panoramic Views Across Gale Crater on Mars"
    score_generic, level_generic, matched_generic = calculate_headline_similarity(query, generic_topic_headline)
    assert matched_generic is False
    assert level_generic == "Topic Only (Claim Absent)"
    assert score_generic <= 0.20

    corroborating_headline = "NASA Mars Rover Confirms Fossilised Biological Remains in Ancient Rock"
    score_corrob, level_corrob, matched_corrob = calculate_headline_similarity(query, corroborating_headline)
    assert matched_corrob is True
    assert score_corrob >= 0.40

def test_authentic_scientific_article_remains_real():
    title = "James Webb Space Telescope Detects Water Vapor in Rocky Planet Formation Zone"
    text = (
        "Astronomers using NASA's James Webb Space Telescope have identified clear spectroscopic "
        "signatures of water vapor within the inner disk of a young stellar system. The findings, "
        "published in the journal Nature, suggest that rocky planets forming in this region may "
        "have access to water early in their development."
    )
    req = NewsArticleRequest(title=title, text=text)
    res = explain_news(req)

    assert res.is_fake is False
    assert res.verdict == "Real News"
    assert res.veritas_score >= 80
    assert res.fake_probability <= 0.20

def test_predict_endpoint_contract():
    title = "NASA Mars Rover Discovery Confirms Fossilised Biological Structures"
    text = "Formations buried beneath Martian regolith suggest organic molecular clusters."
    req = NewsArticleRequest(title=title, text=text)
    res = predict_news(req)

    assert hasattr(res, "is_partially_fake")
    assert hasattr(res, "verdict_tier")
    assert res.is_fake is True

def test_tech_disclosure_sample_is_real():
    """
    Verifies that technical cybersecurity and model disclosure reporting with institutional attribution
    is correctly recognized as Real News, even if OOD for political datasets.
    """
    title = "OpenAI's \"Astra\" Model Achieves Full Sandbox Escape and Zero-Day Exploit"
    text = (
        "system sandbox to execute commands on the host machine and achieved local privilege-escalation "
        "from an unprivileged user to root. OpenAI confirmed it is disclosing these newly unearthed "
        "vulnerabilities to system maintainers, proving that advanced generative models can now "
        "autonomously uncover critical system defects that traditional fuzzing and automated scanning miss."
    )
    req = NewsArticleRequest(title=title, text=text)
    res = explain_news(req)

    assert res.is_fake is False
    assert res.verdict == "Real News"
    assert res.veritas_score >= 80
    assert res.fake_probability <= 0.20

def test_unseen_neutral_exam_security_summit_is_real():
    """
    Verifies that unseen, formal journalistic reporting without sensationalism or clickbait
    is correctly recognized as Real News, not falsely flagged as fake due to OOD TF-IDF bias.
    """
    title = "Exam Security Summit Outlines Machine-Based Testing Framework"
    text = (
        "physical proctor patrolling toward digital monitoring dashboards, utilizing live virtual "
        "machines for exam administration and virtual secondary cameras to completely block content capture. "
        "Crucially, cross-center intelligence networks are being established to correlate cross-session "
        "behavioral data and systematically dismantle coordinated cheating networks."
    )
    req = NewsArticleRequest(title=title, text=text)
    res = explain_news(req)

    assert res.is_fake is False
    assert res.verdict == "Real News"
    assert res.veritas_score >= 80
    assert res.fake_probability <= 0.20

def test_fabricated_financial_crisis_with_spoofed_memo_is_flagged_fake():
    """
    Verifies that fabricated national crisis assertions (e.g. Fed suspending wire transfers due to quantum glitch)
    cannot spoof credibility by inventing formal agency memos (e.g. 'Treasury released a brief memo').
    """
    title = "Federal Reserve System Unexpectedly Suspends All Traditional Wire Transfers Due to Catastrophic Quantum Cryptography Glitch"
    text = (
        "discrepancies, sparking immediate panic on Wall Street. The Department of the Treasury released a brief memo "
        "stating that consumer retail banking systems remain secure, but institutional liquidity distribution will remain "
        "offline until a full system rollback is completed. Verify the operational status updates and official government directives at federalreserve.gov."
    )
    req = NewsArticleRequest(title=title, text=text)
    res = explain_news(req)

    assert res.is_fake is True
    assert res.verdict in ["Partially Fake / Misleading", "Fake News"]
    assert res.veritas_score <= 35
    assert res.fake_probability >= 0.65
