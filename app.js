// Backend API URL: Auto-detect local development vs cloud production
const isLocal = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1' || window.location.port === '8000' || window.location.protocol === 'file:';
const BACKEND_API_URL = (isLocal && window.location.protocol !== 'file:') ? window.location.origin : (window.location.protocol === 'file:' ? 'http://127.0.0.1:8000' : 'https://ai-powered-fake-news-detection-bcbb.onrender.com');

let currentActiveTab = 'text';
let lastAnalyzedResult = null;
let currentAbortController = null;

document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("analyzeForm");
    const titleInput = document.getElementById("articleTitle");
    const textInput = document.getElementById("articleText");
    const urlInput = document.getElementById("articleUrl");

    form.addEventListener("submit", async (e) => {
        e.preventDefault();
        await analyzeArticle();
    });

    const handleInputChange = () => {
        const title = titleInput.value.trim();
        const text = textInput.value.trim();
        const url = urlInput ? urlInput.value.trim() : '';

        if (!title && !text && !url) {
            if (currentAbortController) {
                currentAbortController.abort();
                currentAbortController = null;
            }
            document.getElementById("placeholderState").style.display = "block";
            document.getElementById("activeResults").style.display = "none";
            const actions = document.getElementById("resultActions");
            if (actions) actions.style.display = "none";
            resetSubmitButton();
        } else {
            resetSubmitButton();
        }
    };

    titleInput.addEventListener("input", handleInputChange);
    textInput.addEventListener("input", handleInputChange);
    if (urlInput) urlInput.addEventListener("input", handleInputChange);

    checkBackendHealth();
    setInterval(checkBackendHealth, 30000);
});

function switchInputTab(tab) {
    currentActiveTab = tab;
    const tabText = document.getElementById("tabText");
    const tabUrl = document.getElementById("tabUrl");
    const textGroup = document.getElementById("textInputGroup");
    const urlGroup = document.getElementById("urlGroup");

    if (tab === 'url') {
        tabUrl.classList.add("active");
        tabText.classList.remove("active");
        urlGroup.style.display = "block";
        textGroup.style.display = "none";
    } else {
        tabText.classList.add("active");
        tabUrl.classList.remove("active");
        textGroup.style.display = "block";
        urlGroup.style.display = "none";
    }
}

async function checkBackendHealth() {
    try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 8000);
        const response = await fetch(`${BACKEND_API_URL}/health`, { method: "GET", signal: controller.signal });
        clearTimeout(timeoutId);
        if (response.ok) {
            updateNavStatus(true, "Enterprise Cloud Active");
        } else {
            updateNavStatus(false, "Connecting to Cloud...");
        }
    } catch {
        updateNavStatus(false, "Connecting to Cloud...");
    }
}

function updateNavStatus(isOnline, text) {
    const ind = document.getElementById("navStatusIndicator");
    const txt = document.getElementById("statusText");
    if (ind && txt) {
        ind.className = `status-indicator ${isOnline ? "online" : "offline"}`;
        txt.innerText = text;
    }
}

function resetSubmitButton() {
    const submitBtn = document.getElementById("submitBtn");
    if (submitBtn) {
        submitBtn.disabled = false;
        submitBtn.innerHTML = '<i class="fa-solid fa-wand-magic-sparkles"></i> Run Deep Verification';
    }
}

function clearForm() {
    if (currentAbortController) {
        currentAbortController.abort();
        currentAbortController = null;
    }
    document.getElementById("articleTitle").value = "";
    document.getElementById("articleText").value = "";
    const urlInput = document.getElementById("articleUrl");
    if (urlInput) urlInput.value = "";
    document.getElementById("placeholderState").style.display = "block";
    document.getElementById("activeResults").style.display = "none";
    const actions = document.getElementById("resultActions");
    if (actions) actions.style.display = "none";
    lastAnalyzedResult = null;
    resetSubmitButton();
}

async function analyzeArticle() {
    const submitBtn = document.getElementById("submitBtn");
    submitBtn.disabled = true;

    try {
        let endpoint = `${BACKEND_API_URL}/explain`;
        let payload = {};

        if (currentActiveTab === 'url') {
            const url = document.getElementById("articleUrl").value.trim();
            if (!url) {
                alert("Please enter a valid news URL.");
                resetSubmitButton();
                return;
            }
            endpoint = `${BACKEND_API_URL}/api/v1/analyze-url`;
            payload = { url };
        } else {
            const title = document.getElementById("articleTitle").value.trim();
            const text = document.getElementById("articleText").value.trim();
            if (!title && !text) {
                alert("Please enter a headline or article body text to analyze.");
                resetSubmitButton();
                return;
            }
            payload = { title, text };
        }

        const maxAttempts = 3;
        for (let attempt = 1; attempt <= maxAttempts; attempt++) {
            submitBtn.innerHTML = attempt === 1
                ? '<i class="fa-solid fa-spinner fa-spin"></i> Analyzing Veracity...'
                : `<i class="fa-solid fa-spinner fa-spin"></i> Waking Engine (${attempt}/${maxAttempts})...`;

            try {
                currentAbortController = new AbortController();
                const timeoutId = setTimeout(() => {
                    if (currentAbortController) currentAbortController.abort();
                }, 35000);

                const response = await fetch(endpoint, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(payload),
                    signal: currentAbortController.signal
                });
                clearTimeout(timeoutId);

                if (!response.ok) {
                    const errJson = await response.json().catch(() => ({}));
                    throw new Error(errJson.detail || `HTTP Error ${response.status}`);
                }

                const data = await response.json();
                lastAnalyzedResult = data;
                updateNavStatus(true, "Enterprise Cloud Active");
                renderResults(data);
                return;

            } catch (err) {
                if (err.name === 'AbortError') {
                    console.log("Analysis cancelled.");
                    return;
                }
                console.warn(`Attempt ${attempt} failed:`, err.message);
                if (attempt === maxAttempts) {
                    alert(`Engine response: ${err.message || 'Server timeout. Please try again.'}`);
                } else {
                    await new Promise(r => setTimeout(r, 4000));
                }
            }
        }
    } finally {
        currentAbortController = null;
        resetSubmitButton();
    }
}

function renderResults(data) {
    document.getElementById("placeholderState").style.display = "none";
    document.getElementById("activeResults").style.display = "block";
    const actions = document.getElementById("resultActions");
    if (actions) actions.style.display = "flex";

    const isFake = data.is_fake;
    const banner = document.getElementById("verdictBanner");
    const icon = document.getElementById("verdictIcon");
    const title = document.getElementById("verdictTitle");
    const trustScoreNum = document.getElementById("trustScoreNum");
    const progressFill = document.getElementById("progressFill");
    const meterPercent = document.getElementById("meterPercent");

    banner.className = `verdict-banner ${isFake ? "fake" : "real"}`;
    icon.innerHTML = isFake 
        ? '<i class="fa-solid fa-triangle-exclamation"></i>' 
        : '<i class="fa-solid fa-circle-check"></i>';
    title.innerText = isFake ? "Fake News / Disinformation" : "Real / Authentic Journalism";

    const trustScore = data.veritas_score !== undefined ? data.veritas_score : (isFake ? 15 : 95);
    trustScoreNum.innerText = trustScore;
    trustScoreNum.style.color = isFake ? "var(--fake-accent)" : "var(--real-accent)";

    progressFill.className = `progress-fill ${isFake ? "fake" : "real"}`;
    progressFill.style.width = `${data.confidence_percentage}%`;
    meterPercent.innerText = `${isFake ? "Fake Probability" : "Authenticity"}: ${data.confidence_percentage}%`;

    // Publisher & Domain Credibility
    const pubCard = document.getElementById("publisherCard");
    if (data.domain_credibility) {
        pubCard.style.display = "block";
        document.getElementById("pubDomain").innerText = data.domain_credibility.domain;
        document.getElementById("pubType").innerText = data.domain_credibility.publisher_type;
        document.getElementById("pubAuthority").innerText = `${data.domain_credibility.authority_score}/100`;
    } else {
        pubCard.style.display = "none";
    }

    // AI Reasoning & Live Encyclopedic Grounding
    const rationaleText = document.getElementById("rationaleText");
    rationaleText.innerText = data.llm_reasoning && data.llm_reasoning.rationale 
        ? data.llm_reasoning.rationale 
        : "Evaluated across multi-domain neural stylometry and contextual feature representations.";

    const citationsDiv = document.getElementById("knowledgeCitations");
    citationsDiv.innerHTML = "";
    if (data.llm_reasoning && data.llm_reasoning.knowledge_corroboration && data.llm_reasoning.knowledge_corroboration.length > 0) {
        data.llm_reasoning.knowledge_corroboration.forEach(k => {
            const chip = document.createElement("div");
            chip.className = "citation-chip";
            chip.innerHTML = `<i class="fa-solid fa-book-bookmark"></i> <div><strong>${k.title}:</strong> ${k.snippet}</div>`;
            citationsDiv.appendChild(chip);
        });
    }

    // Claim-by-Claim Forensic Matrix
    const claimsBlock = document.getElementById("claimsBlock");
    const claimsList = document.getElementById("claimsList");
    claimsList.innerHTML = "";
    if (data.claims_breakdown && data.claims_breakdown.length > 0) {
        claimsBlock.style.display = "block";
        data.claims_breakdown.forEach(c => {
            const item = document.createElement("div");
            item.className = `claim-item ${c.tag_class}`;
            item.innerHTML = `
                <div class="claim-top">
                    <span class="claim-tag ${c.tag_class}">Claim #${c.claim_id}: ${c.category}</span>
                    <span class="small-text">${c.risk_level}</span>
                </div>
                <p>"${c.text}"</p>
                <p class="claim-note">${c.note}</p>
            `;
            claimsList.appendChild(item);
        });
    } else {
        claimsBlock.style.display = "none";
    }

    // Top Salient Token Chips
    const chipsContainer = document.getElementById("chipsContainer");
    chipsContainer.innerHTML = "";
    const indicators = isFake ? data.fake_indicators : data.real_indicators;
    if (indicators && indicators.length > 0) {
        indicators.forEach(item => {
            const chip = document.createElement("span");
            chip.className = `chip ${isFake ? "fake" : "real"}`;
            chip.innerHTML = `<i class="fa-solid fa-${isFake ? 'flag' : 'check'}"></i> ${item.token} <strong>(+${item.weight})</strong>`;
            chipsContainer.appendChild(chip);
        });
    } else {
        chipsContainer.innerHTML = '<span class="small-text">Balanced stylistic syntax.</span>';
    }

    // Highlighted Text View
    const annotatedBox = document.getElementById("annotatedBox");
    annotatedBox.innerHTML = data.highlighted_html || "<p>Annotation rendered.</p>";
}

function exportForensicReport() {
    if (!lastAnalyzedResult) return;
    const jsonStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(lastAnalyzedResult, null, 2));
    const dlAnchor = document.createElement('a');
    dlAnchor.setAttribute("href", jsonStr);
    dlAnchor.setAttribute("download", `VeritasAI_Forensic_Report_${Date.now()}.json`);
    document.body.appendChild(dlAnchor);
    dlAnchor.click();
    dlAnchor.remove();
}

function openFeedbackModal() {
    document.getElementById("feedbackModal").style.display = "flex";
}

function closeFeedbackModal() {
    document.getElementById("feedbackModal").style.display = "none";
}

async function submitFeedbackForm() {
    const verdict = document.getElementById("feedbackVerdict").value;
    const notes = document.getElementById("feedbackNotes").value;
    const title = document.getElementById("articleTitle").value || (lastAnalyzedResult ? lastAnalyzedResult.verdict : "");
    const text = document.getElementById("articleText").value || "";

    try {
        await fetch(`${BACKEND_API_URL}/api/v1/feedback`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                title: title,
                text: text,
                predicted_verdict: lastAnalyzedResult ? lastAnalyzedResult.verdict : "Unknown",
                user_reported_verdict: verdict,
                notes: notes
            })
        });
        alert("Thank you! Feedback recorded for model active learning retraining.");
        closeFeedbackModal();
    } catch {
        alert("Failed to submit feedback.");
    }
}
