// Backend API URL: Auto-detect local development vs cloud production
const isLocal = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1' || window.location.port === '8000' || window.location.protocol === 'file:';
const BACKEND_API_URL = (isLocal && window.location.protocol !== 'file:') ? window.location.origin : (window.location.protocol === 'file:' ? 'http://127.0.0.1:8000' : 'https://ai-powered-fake-news-detection-bcbb.onrender.com');

let currentActiveTab = 'text';
let lastAnalyzedResult = null;
let currentAbortController = null;

function escapeHtml(str) {
    if (!str) return "";
    return String(str)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}

const DEMO_PRESETS = {
    nasa: {
        title: "NASA James Webb Space Telescope Detects Water Vapor in Planet-Forming Zone",
        text: "Astronomers using NASA's James Webb Space Telescope have identified clear spectroscopic signatures of water vapor within the inner disk of a young stellar system. The findings, published in the journal Nature, suggest that rocky planets forming in this region may have access to water early in their development."
    },
    cure: {
        title: "MIRACLE CURE: Secret Ancient Root Cures All Disease Overnight [MUST SEE]",
        text: "Doctors are STUNNED and pharmaceutical executives are in a panic! This 100% natural ancient herbal remedy is being suppressed because it completely cures every disease in just 24 hours. The medical establishment does not want you to know the truth!"
    },
    fed: {
        title: "Federal Reserve Holds Benchmark Interest Rates Steady Amid Stable Economic Growth",
        text: "The Federal Reserve announced on Wednesday that it will maintain its benchmark interest rate within the current target range following a unanimous vote by the Federal Open Market Committee. Central bank officials cited continuing job growth and steady consumer spending in their official statement."
    }
};

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

function loadSample(key) {
    const sample = DEMO_PRESETS[key];
    if (!sample) return;
    
    switchInputTab('text');
    document.getElementById("articleTitle").value = sample.title;
    document.getElementById("articleText").value = sample.text;
    analyzeArticle();
}

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
            updateNavStatus(true, "AI Fact-Checker Ready");
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
        submitBtn.innerHTML = '<i class="fa-solid fa-shield-check"></i> Verify This News';
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
                alert("Please enter a news article web link (URL).");
                resetSubmitButton();
                return;
            }
            endpoint = `${BACKEND_API_URL}/api/v1/analyze-url`;
            payload = { url };
        } else {
            const title = document.getElementById("articleTitle").value.trim();
            const text = document.getElementById("articleText").value.trim();
            if (!title && !text) {
                alert("Please enter a news headline or article story to check.");
                resetSubmitButton();
                return;
            }
            payload = { title, text };
        }

        const maxAttempts = 3;
        for (let attempt = 1; attempt <= maxAttempts; attempt++) {
            submitBtn.innerHTML = attempt === 1
                ? '<i class="fa-solid fa-spinner fa-spin"></i> Checking Facts & Credibility...'
                : `<i class="fa-solid fa-spinner fa-spin"></i> Waking AI Engine (${attempt}/${maxAttempts})...`;

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
                updateNavStatus(true, "AI Fact-Checker Ready");
                renderResults(data);
                return;

            } catch (err) {
                if (err.name === 'AbortError') {
                    console.log("Analysis cancelled.");
                    return;
                }
                console.warn(`Attempt ${attempt} failed:`, err.message);
                if (attempt === maxAttempts) {
                    alert(`We could not complete the check: ${err.message || 'Server timeout. Please try again.'}`);
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
    const isPartiallyFake = data.is_partially_fake || data.verdict_tier === 'partially_fake';
    const banner = document.getElementById("verdictBanner");
    const icon = document.getElementById("verdictIcon");
    const title = document.getElementById("verdictTitle");
    const trustScoreNum = document.getElementById("trustScoreNum");
    const scoreLabelText = document.getElementById("scoreLabelText");

    if (isPartiallyFake) {
        banner.className = "verdict-banner mixed";
        icon.innerHTML = '<i class="fa-solid fa-triangle-exclamation"></i>';
        title.innerText = "Caution: Partially Fake or Unverified Claims";
        scoreLabelText.innerText = "High Veracity Risk";
    } else if (isFake) {
        banner.className = "verdict-banner fake";
        icon.innerHTML = '<i class="fa-solid fa-triangle-exclamation"></i>';
        title.innerText = "Warning: Likely Fake or Misleading";
        scoreLabelText.innerText = "High Risk Rating";
    } else {
        banner.className = "verdict-banner real";
        icon.innerHTML = '<i class="fa-solid fa-circle-check"></i>';
        title.innerText = "Verified: Looks Real & Credible";
        scoreLabelText.innerText = "High Credibility";
    }

    const trustScore = data.veritas_score !== undefined ? data.veritas_score : (isFake ? 12 : 95);
    trustScoreNum.innerText = trustScore;

    // Extracted Web Article Metadata
    const urlCard = document.getElementById("urlMetadataCard");
    if (data.extracted_metadata && data.extracted_metadata.extracted_title) {
        if (urlCard) {
            urlCard.style.display = "block";
            const domainEl = document.getElementById("urlMetaDomain");
            if (domainEl) domainEl.innerText = data.extracted_metadata.domain || "";
            const titleEl = document.getElementById("urlMetaTitle");
            if (titleEl) titleEl.innerText = data.extracted_metadata.extracted_title;
            
            const chips = [];
            if (data.extracted_metadata.author) {
                chips.push(`<span class="url-meta-chip"><i class="fa-solid fa-user-pen"></i> ${escapeHtml(data.extracted_metadata.author)}</span>`);
            }
            if (data.extracted_metadata.published_date) {
                chips.push(`<span class="url-meta-chip"><i class="fa-solid fa-calendar"></i> ${escapeHtml(data.extracted_metadata.published_date.slice(0, 10))}</span>`);
            }
            if (data.extracted_metadata.word_count) {
                chips.push(`<span class="url-meta-chip"><i class="fa-solid fa-file-lines"></i> ${data.extracted_metadata.word_count} words</span>`);
            }
            if (data.extracted_metadata.reading_time_min) {
                chips.push(`<span class="url-meta-chip"><i class="fa-solid fa-clock"></i> ~${data.extracted_metadata.reading_time_min}m read</span>`);
            }
            const chipsEl = document.getElementById("urlMetaChips");
            if (chipsEl) chipsEl.innerHTML = chips.join("");
        }
    } else {
        if (urlCard) urlCard.style.display = "none";
    }

    // Publisher & Source Reputation
    const pubCard = document.getElementById("publisherCard");
    if (data.domain_credibility && data.domain_credibility.domain !== "Unknown Source") {
        pubCard.style.display = "block";
        document.getElementById("pubDomain").innerText = data.domain_credibility.domain;
        document.getElementById("pubType").innerText = "Publisher Check";
        document.getElementById("pubAuthority").innerText = data.domain_credibility.publisher_type;
    } else {
        pubCard.style.display = "none";
    }

    // AI Reasoning & Live Knowledge
    const rationaleText = document.getElementById("rationaleText");
    rationaleText.innerText = data.llm_reasoning && data.llm_reasoning.rationale 
        ? data.llm_reasoning.rationale 
        : "Evaluation based on writing style, source attribution, and factual patterns.";

    const citationsDiv = document.getElementById("knowledgeCitations");
    citationsDiv.innerHTML = "";
    if (data.llm_reasoning && data.llm_reasoning.knowledge_corroboration && data.llm_reasoning.knowledge_corroboration.length > 0) {
        data.llm_reasoning.knowledge_corroboration.forEach(k => {
            const chip = document.createElement("div");
            chip.className = "citation-chip";
            chip.innerHTML = `<i class="fa-solid fa-book-open"></i> <div><strong>${escapeHtml(k.title)}:</strong> ${escapeHtml(k.snippet)}</div>`;
            citationsDiv.appendChild(chip);
        });
    }

    // Live News Wire Grounding & Cross-Check
    const newsWireCard = document.getElementById("newsWireCard");
    const newsWireList = document.getElementById("newsWireList");
    const wireStatusBadge = document.getElementById("wireStatusBadge");
    
    if (newsWireCard && newsWireList && wireStatusBadge) {
        const wireArticles = data.news_corroboration || (data.llm_reasoning && data.llm_reasoning.news_corroboration) || [];
        const hasWire = data.has_wire_corroboration || (data.llm_reasoning && data.llm_reasoning.has_wire_corroboration);
        
        if (wireArticles.length > 0) {
            newsWireCard.style.display = "block";
            if (hasWire) {
                wireStatusBadge.className = "wire-status-badge verified";
                wireStatusBadge.innerHTML = '<i class="fa-solid fa-shield-halved"></i> Wire Verified';
            } else {
                wireStatusBadge.className = "wire-status-badge info";
                wireStatusBadge.innerHTML = `<i class="fa-solid fa-newspaper"></i> ${wireArticles.length} News Reports`;
            }
            
            newsWireList.innerHTML = wireArticles.map(article => {
                const isWire = article.is_wire_source;
                const matchScorePct = Math.round(article.match_score * 100);
                const matchClass = article.match_score >= 0.4 ? "high" : (article.match_score >= 0.2 ? "moderate" : "contextual");
                
                return `
                    <div class="news-wire-item">
                        <div class="news-wire-top">
                            <span class="wire-source-tag ${isWire ? 'wire-outlet' : ''}">
                                <i class="fa-solid ${isWire ? 'fa-certificate' : 'fa-bullhorn'}"></i> ${escapeHtml(article.source)}
                            </span>
                            <span class="wire-match-badge ${matchClass}">
                                ${matchScorePct > 0 ? matchScorePct + '% Match' : escapeHtml(article.match_level)}
                            </span>
                        </div>
                        <a href="${article.link || '#'}" target="_blank" rel="noopener noreferrer" class="wire-headline-link">
                            ${escapeHtml(article.title)} <i class="fa-solid fa-arrow-up-right-from-square fa-xs"></i>
                        </a>
                        ${article.pub_date ? `<span class="wire-date"><i class="fa-regular fa-clock"></i> ${escapeHtml(article.pub_date)}</span>` : ''}
                    </div>
                `;
            }).join("");
        } else if (data.topic_covered_claim_absent) {
            newsWireCard.style.display = "block";
            wireStatusBadge.className = "wire-status-badge unverified";
            wireStatusBadge.innerHTML = '<i class="fa-solid fa-triangle-exclamation"></i> Uncorroborated Claim';
            newsWireList.innerHTML = `
                <div class="empty-state" style="padding: 12px; font-size: 0.82rem; color: var(--text-muted); text-align: center;">
                    <p><i class="fa-solid fa-satellite-dish"></i> While the topic entities are actively covered in current press wires, zero verified wire services report or confirm this specific breakthrough claim.</p>
                </div>
            `;
        } else if (isFake) {
            newsWireCard.style.display = "block";
            wireStatusBadge.className = "wire-status-badge unverified";
            wireStatusBadge.innerHTML = '<i class="fa-solid fa-triangle-exclamation"></i> 0 Wire Reports';
            newsWireList.innerHTML = `
                <div class="empty-state" style="padding: 12px; font-size: 0.82rem; color: var(--text-muted); text-align: center;">
                    <p><i class="fa-solid fa-satellite-dish"></i> No corroborating breaking news or press wire coverage was found for this headline across major international news networks.</p>
                </div>
            `;
        } else {
            newsWireCard.style.display = "none";
        }
    }

    // Sentence-by-Sentence Breakdown
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
                    <span class="claim-tag ${c.tag_class}">${c.category}</span>
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

    // Top Salient Keywords (clean labels without math)
    const chipsContainer = document.getElementById("chipsContainer");
    chipsContainer.innerHTML = "";
    const indicators = isFake ? data.fake_indicators : data.real_indicators;
    if (indicators && indicators.length > 0) {
        indicators.forEach(item => {
            const chip = document.createElement("span");
            chip.className = `chip ${isFake ? "fake" : "real"}`;
            chip.innerHTML = `<i class="fa-solid fa-${isFake ? 'flag' : 'check'}"></i> ${item.token}`;
            chipsContainer.appendChild(chip);
        });
    } else {
        chipsContainer.innerHTML = '<span class="small-text">No unusual keyword triggers detected.</span>';
    }

    // Highlighted Text View
    const annotatedBox = document.getElementById("annotatedBox");
    annotatedBox.innerHTML = data.highlighted_html || "<p>Annotated story.</p>";
}

function exportForensicReport() {
    if (!lastAnalyzedResult) return;
    const jsonStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(lastAnalyzedResult, null, 2));
    const dlAnchor = document.createElement('a');
    dlAnchor.setAttribute("href", jsonStr);
    dlAnchor.setAttribute("download", `Veritas_Verification_Report_${Date.now()}.json`);
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
        alert("Thank you! Your feedback helps our AI get smarter.");
        closeFeedbackModal();
    } catch {
        alert("Failed to submit feedback.");
    }
}
