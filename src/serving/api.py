import os
import json
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np

from src.config import config
from src.data.preprocessor import fuse_title_body, extract_stylistic_features
from src.models.baselines import FakeNewsPipeline
from src.explainability.token_saliency import extract_tfidf_word_importance, generate_highlighted_html
from src.llm_reasoner.fact_check_agent import LLMFactCheckReasoner

app = FastAPI(
    title="AI-Powered Fake News Detection API",
    version="1.0.0",
    description="Production REST API (0 = Fake News, 1 = Real News)."
)

# Enable CORS for Netlify frontend and cross-origin clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class NewsArticleRequest(BaseModel):
    title: Optional[str] = ""
    text: Optional[str] = ""

class BatchNewsRequest(BaseModel):
    articles: List[NewsArticleRequest]

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
    fake_indicators: List[dict]
    real_indicators: List[dict]
    highlighted_html: str
    llm_reasoning: dict

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

def compute_hybrid_fake_probability(raw_model_fake_proba: float, stylistic_info: dict) -> float:
    """
    Calibrates statistical n-gram probabilities against structural, capitalization,
    and sensationalist risk factors to avoid entity-shortcut false positives.
    """
    risk = stylistic_info.get("stylistic_fake_risk", 0.0)
    sensational_score = stylistic_info.get("sensational_score", 0.0)
    attribution_score = stylistic_info.get("attribution_score", 0.0)
    is_all_caps = stylistic_info.get("is_all_caps_title", False) or stylistic_info.get("is_all_caps_body", False)

    # Base statistical probability
    p_fake = raw_model_fake_proba

    # High stylistic risk / all-caps shouting / clickbait triggers
    if risk >= 0.40 or is_all_caps or sensational_score >= 0.5:
        # Override entity-biased low fake scores
        p_fake = max(p_fake, 0.45 * p_fake + 0.55 * risk)
        if is_all_caps and p_fake < 0.70:
            p_fake = max(p_fake, 0.78 + (risk * 0.15))
        if sensational_score > 0 and p_fake < 0.65:
            p_fake = max(p_fake, 0.72)
    elif attribution_score >= 0.20 and risk <= 0.20:
        # Journalistic attribution present with zero/low alarmism
        p_fake = max(0.05, p_fake - (attribution_score * 0.40))
        
    return float(np.clip(p_fake, 0.0001, 0.9999))

@app.get("/health")
def health():
    return {"status": "healthy", "service": "Fake News Detection Engine"}

@app.post("/predict", response_model=PredictionResponse)
def predict_news(request: NewsArticleRequest):
    model = get_model()
    fused_text = fuse_title_body(request.title, request.text, title_repeat=1)
    if not fused_text:
        raise HTTPException(status_code=400, detail="Both title and text cannot be empty.")
        
    # Model Target Convention: 0 = Fake News, 1 = Real News
    raw_proba_fake = float(model.predict_proba([fused_text])[0, 0])
    raw_proba_real = float(model.predict_proba([fused_text])[0, 1])
    stylistic_info = extract_stylistic_features(request.title, request.text)
    proba_fake = compute_hybrid_fake_probability(raw_proba_fake, stylistic_info)
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
        
    # Model Target Convention: 0 = Fake News, 1 = Real News
    raw_proba_fake = float(model.predict_proba([fused_text])[0, 0])
    raw_proba_real = float(model.predict_proba([fused_text])[0, 1])
    stylistic_info = extract_stylistic_features(request.title, request.text)
    proba_fake = compute_hybrid_fake_probability(raw_proba_fake, stylistic_info)
    proba_real = 1.0 - proba_fake
    
    is_fake = proba_fake >= 0.5
    confidence = proba_fake if is_fake else proba_real
    verdict = "Fake News" if is_fake else "Real News"
    
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
    
    return ExplainablePredictionResponse(
        verdict=verdict,
        fake_probability=round(proba_fake, 4),
        confidence_percentage=round(confidence * 100, 2),
        is_fake=is_fake,
        fake_indicators=saliency['fake_indicators'],
        real_indicators=saliency['real_indicators'],
        highlighted_html=highlighted_html,
        llm_reasoning=reasoning
    )
