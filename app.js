const PRESETS = {
    economy: {
        title: "Federal Reserve Holds Benchmark Interest Rates Steady Amid Stable Economic Growth",
        text: "The Federal Reserve announced on Wednesday that it will maintain its benchmark interest rate within the current target range following a unanimous vote by the Federal Open Market Committee. Central bank officials cited continuing job growth, steady consumer spending, and moderate inflation figures in their official policy statement released in Washington."
    },
    science: {
        title: "James Webb Space Telescope Detects Water Vapor in Rocky Planet Formation Zone",
        text: "Astronomers using NASAs James Webb Space Telescope have identified clear spectroscopic signatures of water vapor within the inner disk of a young stellar system. The findings, published in the journal Nature, suggest that rocky exoplanets forming in this region may have access to a substantial reservoir of water early in their development."
    },
    conspiracy: {
        title: "SHOCKING BOMBSHELL: Secret Globalist Plot Leaked To Ban All Cash And Confiscate Savings By Next Week [VIDEO]",
        text: "UNBELIEVABLE! Top secret government whistleblowers have exposed an explosive classified document proving corrupt globalist elites are orchestrating a total financial blackout to seize your private bank accounts! Mainstream media refuses to report this terrifying scheme. Watch the emergency video before censors take it down!"
    },
    medical: {
        title: "MIRACLE CURE EXPOSED: Big Pharma Panic As Secret Ancient Root Cures All Disease Overnight [MUST SEE]",
        text: "Doctors are STUNNED and corrupt pharmaceutical executives are in a panic! This 100% natural ancient herbal remedy is being suppressed because it completely reverses aging and cures every chronic condition in just 24 hours. The medical establishment does not want you to know the truth!"
    }
};

// Permanent Live Backend API Endpoint on Render
const BACKEND_API_URL = "https://ai-powered-fake-news-detection-bcbb.onrender.com";

document.addEventListener("DOMContentLoaded", () => {
    // Form submission
    document.getElementById("analyzeForm").addEventListener("submit", async (e) => {
        e.preventDefault();
        await analyzeArticle();
    });

    // Check Backend Status
    checkBackendHealth();
});

async function checkBackendHealth() {
    try {
        const response = await fetch(`${BACKEND_API_URL}/health`, { method: "GET" });
        if (response.ok) {
            updateNavStatus(true, "AI Cloud Engine Active");
        } else {
            updateNavStatus(false, "API Standby / Starting");
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
    submitBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Analyzing with AI...';

    try {
        const response = await fetch(`${BACKEND_API_URL}/explain`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ title, text })
        });

        if (!response.ok) {
            throw new Error(`Server responded with status ${response.status}`);
        }

        const data = await response.json();
        renderResults(data);

    } catch (err) {
        alert(`Note: The backend cloud service may take a moment to wake up if idle (Render free tier). Please try again in 10-20 seconds.\n\nError details: ${err.message}`);
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
    title.innerText = isFake ? "Fake / Clickbait News" : "Real / Factual News";
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
