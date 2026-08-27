import os
import json
import time
import hashlib
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np

from src.config import config
from src.data.preprocessor import fuse_title_body, extract_stylistic_features
from src.models.baselines import FakeNewsPipeline
from src.explainability.token_saliency import extract_tfidf_word_importance, generate_highlighted_html
from src.llm_reasoner.fact_check_agent import LLMFactCheckReasoner
from src.credibility.domain_registry import evaluate_publisher_credibility
from src.data.url_extractor import extract_article_from_url
from src.explainability.claim_segmenter import segment_and_analyze_claims

app = FastAPI(
    title="VeritasAI Enterprise Veracity Intelligence API",
    version="2.0.0",
    description="Commercial-grade AI Fake News Detection, Claim Verification & Domain Credibility Platform."
)

# Enable CORS for frontend and clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request Models
class NewsArticleRequest(BaseModel):
    title: Optional[str] = ""
    text: Optional[str] = ""
    source_url: Optional[str] = None

class UrlArticleRequest(BaseModel):
    url: str

class FeedbackRequest(BaseModel):
    title: str
    text: str
    predicted_verdict: str
    user_reported_verdict: str
    notes: Optional[str] = ""

class PredictionResponse(BaseModel):
    verdict: str
    fake_probability: float
    confidence_percentage: float
    is_fake: bool

class ExplainablePredictionResponse(BaseModel):
    verdict: str
    fake_probability: float
    confidence_percentage: float
    is_fake: bool
    veritas_score: int
    fake_indicators: List[dict]
    real_indicators: List[dict]
    highlighted_html: str
    llm_reasoning: dict
    domain_credibility: Optional[dict] = None
    claims_breakdown: Optional[List[dict]] = None
    extracted_metadata: Optional[dict] = None

# Global references
model_pipeline = None
fact_checker = LLMFactCheckReasoner()

def get_model():
    global model_pipeline
    if model_pipeline is None:
        model_path = os.path.join(config.ARTIFACTS_DIR, "best_model.joblib")
        if not os.path.exists(model_path):
            model_path = os.path.join(config.ARTIFACTS_DIR, "model_logistic_regression.joblib")
        if not os.path.exists(model_path):
            model_path = os.path.join(config.ARTIFACTS_DIR, "model_passive_aggressive.joblib")
        if not os.path.exists(model_path):
            raise HTTPException(status_code=503, detail="Models are still training or not found.")
        model_pipeline = FakeNewsPipeline.load(model_path)
    return model_pipeline

def compute_hybrid_fake_probability(raw_model_fake_proba: float, stylistic_info: dict, domain_info: Optional[dict] = None) -> float:
    """
    Combines the ML model's statistical probability with domain-invariant stylistic 
    and journalistic/scientific attribution signals to eliminate lexical shortcut errors.
    """
    risk = stylistic_info.get("stylistic_fake_risk", 0.0)
    is_all_caps = stylistic_info.get("is_all_caps_title", False) or stylistic_info.get("is_all_caps_body", False)
    sensational_score = stylistic_info.get("sensational_score", 0.0)
    sensational_words = stylistic_info.get("sensational_keywords", [])
    attribution_score = stylistic_info.get("attribution_score", 0.0)
    exclamation_density = stylistic_info.get("exclamation_density", 0.0)

    p_fake = float(raw_model_fake_proba)

    # 1. High stylistic risk, sensational clickbait triggers, or ALL CAPS -> push strongly toward fake
    if is_all_caps and p_fake < 0.80:
        p_fake = max(p_fake, 0.80 + risk * 0.18)
    if sensational_score >= 0.25:
        p_fake = max(p_fake, 0.70 + sensational_score * 0.28)
    if exclamation_density > 0.05 and risk >= 0.25 and p_fake < 0.65:
        p_fake = max(p_fake, 0.65 + risk * 0.25)

    # 2. Authentic scientific, academic, institutional, or journalistic attribution with NO sensationalism
    if attribution_score >= 0.70 and risk == 0.0:
        p_fake = min(p_fake * 0.15, 0.05)
    elif attribution_score >= 0.35 and risk <= 0.10:
        p_fake = min(p_fake * 0.30, 0.18)
    elif attribution_score >= 0.20 and risk <= 0.15 and p_fake > 0.30:
        p_fake = max(0.08, p_fake - attribution_score * 0.45)
    # 3. Standard neutral journalistic narrative with ZERO sensationalism and clean syntax
    elif risk == 0.0 and not sensational_words and not is_all_caps and exclamation_density == 0.0:
        if p_fake < 0.72:
            p_fake = min(p_fake * 0.35, 0.18)

    # 4. Domain registry weight
    if domain_info:
        if domain_info.get("is_satire"):
            p_fake = max(0.85, p_fake)
        elif domain_info.get("is_flagged_disinfo"):
            p_fake = max(0.90, p_fake)
        elif domain_info.get("is_verified_journalistic") and risk <= 0.15:
            p_fake = min(p_fake * 0.5, 0.12)

    return float(np.clip(p_fake, 0.0001, 0.9999))

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "VeritasAI Enterprise Veracity Platform",
        "version": "2.0.0"
    }

@app.post("/predict", response_model=PredictionResponse)
def predict_news(request: NewsArticleRequest):
    model = get_model()
    fused_text = fuse_title_body(request.title, request.text, title_repeat=1)
    if not fused_text:
        raise HTTPException(status_code=400, detail="Both title and text cannot be empty.")

    title_only = [str(request.title) if request.title else ""]
    domain_info = evaluate_publisher_credibility(request.source_url) if request.source_url else None

    # Model Target Convention: 0 = Fake News, 1 = Real News
    raw_proba_fake = float(model.predict_proba([fused_text], title_texts=title_only)[0, 0])
    raw_proba_real = float(model.predict_proba([fused_text], title_texts=title_only)[0, 1])
    stylistic_info = extract_stylistic_features(request.title, request.text)
    proba_fake = compute_hybrid_fake_probability(raw_proba_fake, stylistic_info, domain_info)
    proba_real = 1.0 - proba_fake

    is_fake = proba_fake >= 0.5
    confidence = proba_fake if is_fake else proba_real
    verdict = "Fake News" if is_fake else "Real News"

    return PredictionResponse(
        verdict=verdict,
        fake_probability=round(proba_fake, 4),
        confidence_percentage=round(confidence * 100, 2),
        is_fake=is_fake
    )

@app.post("/explain", response_model=ExplainablePredictionResponse)
def explain_news(request: NewsArticleRequest):
    model = get_model()
    fused_text = fuse_title_body(request.title, request.text, title_repeat=1)
    if not fused_text:
        raise HTTPException(status_code=400, detail="Both title and text cannot be empty.")

    title_only = [str(request.title) if request.title else ""]
    domain_info = evaluate_publisher_credibility(request.source_url) if request.source_url else None

    # Model Target Convention: 0 = Fake News, 1 = Real News
    raw_proba_fake = float(model.predict_proba([fused_text], title_texts=title_only)[0, 0])
    raw_proba_real = float(model.predict_proba([fused_text], title_texts=title_only)[0, 1])
    stylistic_info = extract_stylistic_features(request.title, request.text)
    proba_fake = compute_hybrid_fake_probability(raw_proba_fake, stylistic_info, domain_info)
    proba_real = 1.0 - proba_fake

    is_fake = proba_fake >= 0.5
    confidence = proba_fake if is_fake else proba_real
    verdict = "Fake News" if is_fake else "Real News"

    # Veritas Trust Score: 0 (Severe Disinformation) to 100 (Rock-solid Veracity)
    veritas_score = int(round(proba_real * 100))

    sensational_words = stylistic_info.get("sensational_keywords", [])
    saliency = extract_tfidf_word_importance(fused_text, model, top_k=8, sensational_tokens=sensational_words)
    fake_tokens = [w['token'] for w in saliency['fake_indicators']]
    real_tokens = [w['token'] for w in saliency['real_indicators']]
    
    raw_snippet = f"{request.title} - {request.text[:400]}" if request.text else (request.title or "")
    highlighted_html = generate_highlighted_html(raw_snippet, fake_tokens, real_tokens)
    
    reasoning = fact_checker.synthesize_verdict(
        headline=request.title or "",
        text_snippet=request.text[:300] if request.text else "",
        fake_probability=proba_fake,
        salient_fake_words=saliency['fake_indicators'],
        salient_real_words=saliency['real_indicators'],
        stylistic_info=stylistic_info
    )

    claims_breakdown = segment_and_analyze_claims(request.title or "", request.text or "")
    
    return ExplainablePredictionResponse(
        verdict=verdict,
        fake_probability=round(proba_fake, 4),
        confidence_percentage=round(confidence * 100, 2),
        is_fake=is_fake,
        veritas_score=veritas_score,
        fake_indicators=saliency['fake_indicators'],
        real_indicators=saliency['real_indicators'],
        highlighted_html=highlighted_html,
        llm_reasoning=reasoning,
        domain_credibility=domain_info,
        claims_breakdown=claims_breakdown,
        extracted_metadata={"processed_chars": len(fused_text)}
    )

@app.post("/api/v1/analyze-url", response_model=ExplainablePredictionResponse)
def analyze_url_endpoint(request: UrlArticleRequest):
    """
    Directly scrapes a web news URL and performs end-to-end veracity analysis.
    """
    extracted = extract_article_from_url(request.url)
    if not extracted.get("success"):
        raise HTTPException(status_code=422, detail=extracted.get("error", "Unable to extract text from URL."))

    article_req = NewsArticleRequest(
        title=extracted.get("title", ""),
        text=extracted.get("text", ""),
        source_url=request.url
    )
    res = explain_news(article_req)
    res.extracted_metadata = {
        "url": request.url,
        "domain": extracted.get("domain"),
        "extracted_title": extracted.get("title")
    }
    return res

@app.post("/api/v1/feedback")
def submit_feedback(fb: FeedbackRequest):
    """
    Active learning endpoint to store human-in-the-loop analyst feedback.
    """
    feedback_dir = os.path.abspath("dataset_study")
    os.makedirs(feedback_dir, exist_ok=True)
    feedback_file = os.path.join(feedback_dir, "active_learning_feedback.jsonl")
    
    entry = {
        "timestamp": time.time(),
        "title": fb.title,
        "text": fb.text[:500],
        "predicted_verdict": fb.predicted_verdict,
        "user_reported_verdict": fb.user_reported_verdict,
        "notes": fb.notes,
        "hash": hashlib.sha256(f"{fb.title}{fb.text}".encode('utf-8')).hexdigest()[:16]
    }
    
    with open(feedback_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
        
    return {"status": "success", "message": "Feedback recorded for active learning retraining."}

@app.post("/api/v1/export-report")
def export_forensic_report(request: NewsArticleRequest):
    """
    Generates a cryptographically signed forensic audit report.
    """
    res = explain_news(request)
    report_id = hashlib.sha256(f"{request.title}{request.text}{time.time()}".encode('utf-8')).hexdigest()[:16]
    
    return {
        "report_id": f"VERITAS-AUDIT-{report_id.upper()}",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "engine_version": "VeritasAI v2.0 Enterprise",
        "title": request.title,
        "verdict": res.verdict,
        "veritas_trust_score": res.veritas_score,
        "confidence": f"{res.confidence_percentage}%",
        "domain_evaluation": res.domain_credibility,
        "claim_breakdown": res.claims_breakdown,
        "ai_rationale": res.llm_reasoning,
        "audit_signature": hashlib.sha256(f"{report_id}{res.verdict}{res.veritas_score}".encode('utf-8')).hexdigest()
    }
