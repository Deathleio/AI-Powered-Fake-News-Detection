# Dataset Study 06: Multi-Agent System & LLM Orchestration Architecture

**Project**: AI-Powered Fake News Detection Using NLP & Deep Learning  
**Dataset**: WELFake (`WELFake_Dataset.csv`)  
**Purpose**: Complete architectural blueprint for orchestrating multiple autonomous subagents and LLM reasoning modules with strict token efficiency.

---

## 1. Multi-Agent Topology & Responsibilities

```
                         ┌─────────────────────────────┐
                         │   Lead Orchestrator Agent   │
                         │ (Task Planner & Supervisor) │
                         └──────────────┬──────────────┘
                                        │
     ┌──────────────────┬───────────────┴───────────────┬──────────────────┐
     ▼                  ▼                               ▼                  ▼
┌──────────────┐ ┌──────────────┐                ┌──────────────┐   ┌──────────────┐
│  DataOps     │ │ DeepLearning │                │ Explainable  │   │ LLM Verifier │
│  Subagent    │ │  Subagent    │                │  AI Subagent │   │  & Factcheck │
│              │ │              │                │              │   │   Subagent   │
└──────┬───────┘ └──────┬───────┘                └──────┬───────┘   └──────┬───────┘
       │                │                               │                  │
       └────────────────┴───────────────┬───────────────┴──────────────────┘
                                        ▼
                         ┌─────────────────────────────┐
                         │    Deployment Subagent      │
                         │ (FastAPI & Gradio/Web UI)   │
                         └─────────────────────────────┘
```

---

## 2. Subagent Roles & Contract Interfaces

### 1. `DataOps_Subagent`
- **Role**: Dataset validation, cleaning, leakage removal, Stratified Train/Val/Test creation, TF-IDF / Tokenizer persistence.
- **Output Artifact**: `data_splits_manifest.json`, `preprocessor.pkl`.

### 2. `DeepLearning_Subagent`
- **Role**: Model architecture construction, PyTorch training loops, learning rate scheduling, model checkpointing (`best_model.pt`).
- **Output Artifact**: `training_metrics.json`, `model_weights/`.

### 3. `Explainability_Subagent`
- **Role**: LIME and Integrated Gradients token attribution generation, confusion matrices, ROC curves, and misclassification error audits.
- **Output Artifact**: `xai_report.json`, `attribution_visualizations/`.

### 4. `LLM_FactChecker_Subagent`
- **Role**: Second-opinion reasoning agent. Receives the article snippet + Deep Learning prediction and executes zero-shot chain-of-thought verification.
- **Output Artifact**: Structured JSON explanation for end-user UI.

### 5. `Deployment_Subagent`
- **Role**: Packaging models into a high-throughput REST API (`FastAPI`) and an interactive frontend dashboard (`Gradio`).

---

## 3. Token-Efficient Inter-Agent Communication Protocol

To prevent LLM context exhaustion across agent handoffs:
- **Rule 1**: Never transmit raw CSV data between agents.
- **Rule 2**: Transmit only **Schema-Constrained JSON Summaries** (e.g. metadata, shapes, evaluation metrics, token budgets).

### Standard Inter-Agent JSON Telemetry Format:
```json
{
  "task": "evaluate_model",
  "model_type": "RoBERTa-base",
  "metrics": {
    "accuracy": 0.9842,
    "macro_f1": 0.9839,
    "roc_auc": 0.9971
  },
  "top_saliency_tokens": [
    {"token": "BREAKING", "score": 0.89, "class": 1},
    {"token": "unprecedented", "score": 0.65, "class": 1},
    {"token": "spokesperson", "score": -0.72, "class": 0}
  ],
  "decision": "READY_FOR_DEPLOYMENT"
}
```

---

## 4. LLM Verification Prompt Template (Token-Constrained CoT)

When querying an LLM for real-time fact-checking and explainability synthesis:

```text
[SYSTEM]
You are an expert investigative journalist and fact-checking AI. 
Analyze the provided news headline and excerpt alongside the Deep Learning model's classification score.
Provide a concise, evidence-based verification breakdown in under 120 words.

[INPUT]
Headline: {headline}
Excerpt: {excerpt_max_200_words}
Deep Learning Model Prediction: {model_prediction} ({confidence_pct}% Confidence)
Top Saliency Keywords: {saliency_keywords}

[OUTPUT FORMAT (STRICT JSON)]
{
  "verdict": "Likely Real | Likely Fake | Unverified",
  "confidence_score": 0.0 to 1.0,
  "credibility_indicators": ["list of up to 3 red flags or authenticity markers"],
  "synthesized_reasoning": "Concise 2-sentence rationale explaining the verdict."
}
```

---

## 5. End-to-End Orchestrated Pipeline Flow

1. User submits an article URL or raw text in the UI.
2. `DataOps` sanitizes and tokenizes input.
3. `DeepLearning` generates prediction logits and confidence score.
4. `Explainability` computes top salient word attributions in real time.
5. `LLM Verifier` generates human-readable reasoning without token blowout.
6. `Deployment UI` renders dual gauge meters: Model Score + LLM Reasoning + Highlighted Saliency Span.
