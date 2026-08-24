import os
import uvicorn
from fastapi.responses import HTMLResponse
from src.serving.api import app, NewsArticleRequest, explain_news

HTML_CONTENT = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Powered Fake News Detection System</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        body { background-color: #f8fafc; font-family: 'Segoe UI', system-ui, sans-serif; color: #1e293b; }
        .hero { background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%); color: white; padding: 40px 0 30px; margin-bottom: 30px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); }
        .card { border-radius: 12px; border: none; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05); margin-bottom: 24px; }
        .btn-primary { background-color: #2563eb; border: none; padding: 10px 24px; font-weight: 600; border-radius: 8px; }
        .btn-primary:hover { background-color: #1d4ed8; }
        .badge-fake { background-color: #fee2e2; color: #991b1b; font-weight: 600; padding: 6px 12px; border-radius: 6px; border: 1px solid #fecaca; }
        .badge-real { background-color: #dcfce7; color: #166534; font-weight: 600; padding: 6px 12px; border-radius: 6px; border: 1px solid #bbf7d0; }
        .progress-bar-fake { background-color: #dc2626; }
        .progress-bar-real { background-color: #16a34a; }
        .sample-btn { cursor: pointer; text-decoration: underline; color: #2563eb; font-size: 0.9rem; margin-right: 15px; }
        .sample-btn:hover { color: #1d4ed8; }
        .token-tag { display: inline-block; padding: 3px 8px; margin: 3px; border-radius: 4px; font-size: 0.85rem; font-weight: 500; }
        .token-fake { background-color: #fee2e2; color: #991b1b; border: 1px solid #f87171; }
        .token-real { background-color: #dcfce7; color: #166534; border: 1px solid #4ade80; }
    </style>
</head>
<body>

<div class="hero">
    <div class="container text-center">
        <h1 class="fw-bold mb-2"><i class="fa-solid fa-shield-halved me-2"></i> AI-Powered Fake News Detection</h1>
        <p class="lead mb-0">High-Precision Deep Learning & Explainable NLP Verification Engine</p>
    </div>
</div>

<div class="container">
    <div class="row">
        <!-- Input Column -->
        <div class="col-lg-6">
            <div class="card p-4">
                <h4 class="fw-bold mb-3"><i class="fa-regular fa-newspaper me-2"></i> Analyze Article</h4>
                
                <div class="mb-3">
                    <label class="form-label fw-semibold">News Headline / Title</label>
                    <input type="text" id="newsTitle" class="form-control" placeholder="e.g. U.S. Senate reaches agreement on relief budget..." />
                </div>

                <div class="mb-3">
                    <label class="form-label fw-semibold">Article Content / Body</label>
                    <textarea id="newsText" class="form-control" rows="8" placeholder="Paste full article text or paragraphs here..."></textarea>
                </div>

                <div class="mb-3">
                    <span class="text-muted small fw-semibold me-2">Try Samples:</span>
                    <span class="sample-btn" onclick="loadSample(1)"><i class="fa-solid fa-check me-1"></i> Real News</span>
                    <span class="sample-btn" onclick="loadSample(2)"><i class="fa-solid fa-triangle-exclamation me-1"></i> Fake / Clickbait</span>
                </div>

                <button class="btn btn-primary w-100 py-2" id="analyzeBtn" onclick="analyzeNews()">
                    <i class="fa-solid fa-magnifying-glass-chart me-2"></i> Run AI Analysis
                </button>
            </div>
        </div>

        <!-- Output Column -->
        <div class="col-lg-6">
            <div class="card p-4" id="resultCard" style="display: none;">
                <h4 class="fw-bold mb-3"><i class="fa-solid fa-chart-pie me-2"></i> Classification Results</h4>
                
                <div class="d-flex justify-content-between align-items-center mb-3">
                    <span class="fs-5 fw-bold" id="verdictText">--</span>
                    <span id="verdictBadge" class="fs-6">--</span>
                </div>

                <!-- Confidence Gauge -->
                <label class="form-label small fw-semibold text-muted">Authenticity vs Fake Probability Meter</label>
                <div class="progress mb-3" style="height: 24px;">
                    <div id="probBar" class="progress-bar progress-bar-striped progress-bar-animated fw-bold" role="progressbar" style="width: 0%;">0%</div>
                </div>

                <!-- Saliency Tokens -->
                <div class="mb-3">
                    <h6 class="fw-bold text-muted mb-2"><i class="fa-solid fa-highlighter me-1"></i> Key Saliency Trigger Keywords</h6>
                    <div id="tokensContainer" class="p-2 bg-light rounded border"></div>
                </div>

                <!-- AI Fact Check Rationale -->
                <div class="p-3 bg-light rounded border">
                    <h6 class="fw-bold text-primary mb-1"><i class="fa-solid fa-robot me-1"></i> AI Verification Rationale</h6>
                    <p class="mb-0 text-secondary small" id="reasoningText">--</p>
                </div>
            </div>

            <div class="card p-4 text-center text-muted" id="placeholderCard">
                <i class="fa-solid fa-chart-line fa-3x mb-3 text-secondary opacity-50"></i>
                <h5>Ready for Evaluation</h5>
                <p class="small mb-0">Enter a headline and body or click a sample to see real-time classification, token attributions, and AI reasoning.</p>
            </div>
        </div>
    </div>
</div>

<script>
const SAMPLES = {
    1: {
        title: "Federal Reserve Maintains Benchmark Interest Rate Amid Stable Inflation Data",
        text: "The Federal Reserve announced on Wednesday that it will hold benchmark interest rates steady following a two-day policy meeting. Central bank officials noted that recent economic indicators show moderate growth in employment and household spending, while inflation continues to trend toward the target rate."
    },
    2: {
        title: "SHOCKING BOMBSHELL: Secret Globalist Plot Unveiled To Ban All Cash By Next Week [VIDEO]",
        text: "UNBELIEVABLE! Top secret government insiders have leaked conclusive proof that corrupt elites are secretly orchestrating a total blackout to confiscate private savings. Mainstream media refuses to report this terrifying scheme! Watch the explosive footage before it gets deleted!"
    }
};

function loadSample(id) {
    document.getElementById('newsTitle').value = SAMPLES[id].title;
    document.getElementById('newsText').value = SAMPLES[id].text;
}

async function analyzeNews() {
    const title = document.getElementById('newsTitle').value;
    const text = document.getElementById('newsText').value;
    if (!title && !text) {
        alert("Please enter at least a headline or article body.");
        return;
    }

    const btn = document.getElementById('analyzeBtn');
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span> Analyzing...';

    try {
        const response = await fetch('/explain', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ title, text })
        });

        if (!response.ok) throw new Error("API error");
        const data = await response.json();

        document.getElementById('placeholderCard').style.display = 'none';
        const resultCard = document.getElementById('resultCard');
        resultCard.style.display = 'block';

        const isFake = data.is_fake;
        document.getElementById('verdictText').innerText = data.verdict;
        
        const badge = document.getElementById('verdictBadge');
        badge.className = isFake ? 'badge-fake' : 'badge-real';
        badge.innerText = `${data.confidence_percentage}% Confidence`;

        const bar = document.getElementById('probBar');
        bar.className = `progress-bar progress-bar-striped progress-bar-animated ${isFake ? 'progress-bar-fake' : 'progress-bar-real'}`;
        bar.style.width = `${data.confidence_percentage}%`;
        bar.innerText = `${isFake ? 'Fake Score: ' : 'Authenticity: '} ${data.confidence_percentage}%`;

        // Render Saliency Tokens
        const tokensContainer = document.getElementById('tokensContainer');
        tokensContainer.innerHTML = '';
        if (isFake && data.fake_indicators.length > 0) {
            data.fake_indicators.forEach(item => {
                tokensContainer.innerHTML += `<span class="token-tag token-fake"><i class="fa-solid fa-flag me-1"></i>${item.token} (${item.weight})</span>`;
            });
        } else if (!isFake && data.real_indicators.length > 0) {
            data.real_indicators.forEach(item => {
                tokensContainer.innerHTML += `<span class="token-tag token-real"><i class="fa-solid fa-check me-1"></i>${item.token} (${item.weight})</span>`;
            });
        } else {
            tokensContainer.innerHTML = '<span class="text-muted small">Balanced vocabulary signals.</span>';
        }

        // Rationale
        document.getElementById('reasoningText').innerText = data.llm_reasoning.rationale || "Analysis generated based on trained linguistic weights.";

    } catch (e) {
        alert("Failed to analyze: " + e.message);
    } finally {
        btn.disabled = false;
        btn.innerHTML = '<i class="fa-solid fa-magnifying-glass-chart me-2"></i> Run AI Analysis';
    }
}
</script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def serve_dashboard():
    return HTMLResponse(content=HTML_CONTENT)

def launch_server(host="127.0.0.1", port=8000):
    print(f"Starting Fake News Detection Dashboard on http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)

if __name__ == '__main__':
    launch_server()
