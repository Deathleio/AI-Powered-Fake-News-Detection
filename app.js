const PRESETS = {
    diplomacy: {
        title: "European Parliament Rejects Brexit Proposal Citing Citizen Rights Concerns",
        text: "The European Parliament said on Tuesday that British Prime Minister proposals for European citizens living in Britain fell short and would create a second class of citizens."
    },
    politics: {
        title: "Schumer calls on Trump to appoint official to oversee disaster recovery effort",
        text: "Senate Democratic Leader Chuck Schumer called on President Donald Trump to appoint a single official to manage the recovery effort following the recent hurricane damage."
    },
    conspiracy: {
        title: "SHOCKING BOMBSHELL: Secret Globalist Plot Unveiled To Ban All Cash By Next Week [VIDEO]",
        text: "UNBELIEVABLE! Top secret government insiders have leaked conclusive proof that corrupt elites are secretly orchestrating a total blackout to confiscate private savings. Mainstream media refuses to report this terrifying scheme! Watch the explosive footage before it gets deleted!"
    },
    satire: {
        title: "THE WORLD IS ON FIRE",
        text: "AUSTRALIA BUSHFIRE HAS TAKEN THE LIFE OF TRUMP WHO WAS DANCING WITH NETANYAHU"
    }
};

// Backend API URL: Auto-detect local development vs cloud production
const isLocal = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1' || window.location.port === '8000' || window.location.protocol === 'file:';
const BACKEND_API_URL = (isLocal && window.location.protocol !== 'file:') ? window.location.origin : (window.location.protocol === 'file:' ? 'http://127.0.0.1:8000' : 'https://ai-powered-fake-news-detection-bcbb.onrender.com');

document.addEventListener("DOMContentLoaded", () => {
    document.getElementById("analyzeForm").addEventListener("submit", async (e) => {
        e.preventDefault();
        await analyzeArticle();
    });

    checkBackendHealth();
    setInterval(checkBackendHealth, 30000);
});

async function checkBackendHealth() {
    try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 8000);
        const response = await fetch(`${BACKEND_API_URL}/health`, { method: "GET", signal: controller.signal });
        clearTimeout(timeoutId);
        if (response.ok) {
            updateNavStatus(true, "AI Cloud Engine Active");
        } else {
            updateNavStatus(false, "AI Engine Connecting...");
        }
    } catch {
        updateNavStatus(false, "AI Engine Connecting...");
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

function loadSample(key) {
    if (PRESETS[key]) {
        document.getElementById("articleTitle").value = PRESETS[key].title;
        document.getElementById("articleText").value = PRESETS[key].text;
    }
}

function clearForm() {
    document.getElementById("articleTitle").value = "";
    document.getElementById("articleText").value = "";
    document.getElementById("placeholderState").style.display = "block";
    document.getElementById("activeResults").style.display = "none";
}

async function analyzeArticle() {
    const title = document.getElementById("articleTitle").value.trim();
    const text = document.getElementById("articleText").value.trim();

    if (!title && !text) {
        alert("Please enter a headline or article body text to analyze.");
        return;
    }

    const submitBtn = document.getElementById("submitBtn");
    submitBtn.disabled = true;

    try {
        const maxAttempts = 3;
        for (let attempt = 1; attempt <= maxAttempts; attempt++) {
            submitBtn.innerHTML = attempt === 1
                ? '<i class="fa-solid fa-spinner fa-spin"></i> Analyzing with AI...'
                : `<i class="fa-solid fa-spinner fa-spin"></i> Waking Cloud Engine (${attempt}/${maxAttempts})...`;

            try {
                const controller = new AbortController();
                const timeoutId = setTimeout(() => controller.abort(), 35000);

                const response = await fetch(`${BACKEND_API_URL}/explain`, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ title, text }),
                    signal: controller.signal
                });
                clearTimeout(timeoutId);

                if (!response.ok) {
                    throw new Error(`HTTP Error ${response.status}`);
                }

                const data = await response.json();
                updateNavStatus(true, "AI Cloud Engine Active");
                renderResults(data);
                return; // Success!

            } catch (err) {
                console.warn(`Attempt ${attempt} failed:`, err.message);
                if (attempt === maxAttempts) {
                    alert(`The AI Cloud engine is taking longer than expected to wake up.\n\nPlease wait 10-15 seconds and try clicking 'Run AI Classification' again.`);
                } else {
                    await new Promise(r => setTimeout(r, 4000));
                }
            }
        }
    } finally {
        submitBtn.disabled = false;
        submitBtn.innerHTML = '<i class="fa-solid fa-magnifying-glass-chart"></i> Run AI Classification';
    }
}

function renderResults(data) {
    document.getElementById("placeholderState").style.display = "none";
    document.getElementById("activeResults").style.display = "block";

    const isFake = data.is_fake;
    const banner = document.getElementById("verdictBanner");
    const icon = document.getElementById("verdictIcon");
    const title = document.getElementById("verdictTitle");
    const badge = document.getElementById("confidenceBadge");
    const progressFill = document.getElementById("progressFill");
    const meterPercent = document.getElementById("meterPercent");

    banner.className = `verdict-banner ${isFake ? "fake" : "real"}`;
    icon.innerHTML = isFake 
        ? '<i class="fa-solid fa-triangle-exclamation"></i>' 
        : '<i class="fa-solid fa-circle-check"></i>';
    title.innerText = isFake ? "Fake / Disinformation News" : "Real / Authentic News";
    badge.innerText = `${data.confidence_percentage}% Confidence`;

    progressFill.className = `progress-fill ${isFake ? "fake" : "real"}`;
    progressFill.style.width = `${data.confidence_percentage}%`;
    meterPercent.innerText = `${isFake ? "Fake" : "Authenticity"}: ${data.confidence_percentage}%`;

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
        chipsContainer.innerHTML = '<span class="small-text">Balanced vocabulary patterns.</span>';
    }

    const rationaleText = document.getElementById("rationaleText");
    rationaleText.innerText = data.llm_reasoning && data.llm_reasoning.rationale 
        ? data.llm_reasoning.rationale 
        : "Classification computed via high-dimensional linguistic feature weights.";

    const annotatedBox = document.getElementById("annotatedBox");
    annotatedBox.innerHTML = data.highlighted_html || "<p>Annotation rendered.</p>";
}
