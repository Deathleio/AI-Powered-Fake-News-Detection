import os
import json
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np

from src.config import config
from src.data.preprocessor import fuse_title_body
from src.models.baselines import FakeNewsPipeline
from src.explainability.token_saliency import extract_tfidf_word_importance, generate_highlighted_html
from src.llm_reasoner.fact_check_agent import LLMFactCheckReasoner

app = FastAPI(
    title="AI-Powered Fake News Detection API",
    version="1.0.0",
    description="Production REST API for Real-Time Fake News Detection and Token-Level Explainability."
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

@app.get("/health")
def health():
    return {"status": "healthy", "service": "Fake News Detection Engine"}

@app.post("/predict", response_model=PredictionResponse)
def predict_news(request: NewsArticleRequest):
    model = get_model()
    fused_text = fuse_title_body(request.title, request.text, title_repeat=1)
    if not fused_text:
        raise HTTPException(status_code=400, detail="Both title and text cannot be empty.")
        
    proba = float(model.predict_proba([fused_text])[0, 1])
    is_fake = proba >= 0.5
    confidence = proba if is_fake else (1.0 - proba)
    verdict = "Fake News" if is_fake else "Real News"
    
    return PredictionResponse(
        verdict=verdict,
        fake_probability=round(proba, 4),
        confidence_percentage=round(confidence * 100, 2),
        is_fake=is_fake
    )

@app.post("/explain", response_model=ExplainablePredictionResponse)
def explain_news(request: NewsArticleRequest):
    model = get_model()
    fused_text = fuse_title_body(request.title, request.text, title_repeat=1)
    if not fused_text:
        raise HTTPException(status_code=400, detail="Both title and text cannot be empty.")
        
    proba = float(model.predict_proba([fused_text])[0, 1])
    is_fake = proba >= 0.5
    confidence = proba if is_fake else (1.0 - proba)
    verdict = "Fake News" if is_fake else "Real News"
    
    saliency = extract_tfidf_word_importance(fused_text, model, top_k=8)
    fake_tokens = [w['token'] for w in saliency['fake_indicators']]
    real_tokens = [w['token'] for w in saliency['real_indicators']]
    
    raw_snippet = f"{request.title} - {request.text[:400]}"
    highlighted_html = generate_highlighted_html(raw_snippet, fake_tokens, real_tokens)
    
    reasoning = fact_checker.synthesize_verdict(
        headline=request.title or "",
        text_snippet=request.text[:300] if request.text else "",
        fake_probability=proba,
        salient_fake_words=saliency['fake_indicators'],
        salient_real_words=saliency['real_indicators']
    )
    
    return ExplainablePredictionResponse(
        verdict=verdict,
        fake_probability=round(proba, 4),
        confidence_percentage=round(confidence * 100, 2),
        is_fake=is_fake,
        fake_indicators=saliency['fake_indicators'],
        real_indicators=saliency['real_indicators'],
        highlighted_html=highlighted_html,
        llm_reasoning=reasoning
    )
