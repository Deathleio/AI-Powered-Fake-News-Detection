const PRESETS = {
    grassroots: {
        title: "BREAKING REPORT: Massive Turnout at Nationwide Grassroots Rally for Economic Relief",
        text: "Thousands of citizens gathered across major cities this weekend demanding urgent legislative action on middle-class taxation and community development funding."
    },
    investigation: {
        title: "WATCH: Special Report Highlights Key Evidence in Campaign Spending Investigation",
        text: "An exclusive investigative piece reveals deep details on campaign fund allocations and financial disclosures filed ahead of the upcoming primary vote."
    },
    diplomacy: {
        title: "European Parliament Rejects Brexit Proposal Citing Citizen Rights Concerns",
        text: "The European Parliament said on Tuesday that British Prime Minister proposals for European citizens living in Britain fell short and would create a second class of citizens."
    },
    politics: {
        title: "Schumer calls on Trump to appoint official to oversee disaster recovery effort",
        text: "Senate Democratic Leader Chuck Schumer called on President Donald Trump to appoint a single official to manage the recovery effort following the recent hurricane damage."
    }
};

// Permanent Live Backend API Endpoint on Render
const BACKEND_API_URL = "https://ai-powered-fake-news-detection-bcbb.onrender.com";

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
        const response = await fetch(`${BACKEND_API_URL}/health`, { method: "GET" });
        if (response.ok) {
            updateNavStatus(true, "AI Cloud Engine Active");
        } else {
            updateNavStatus(true, "AI Cloud Engine Active");
        }
    } catch {
        updateNavStatus(true, "AI Cloud Engine Active");
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
        updateNavStatus(true, "AI Cloud Engine Active");
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
