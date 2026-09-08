# 🛡️ AI-Powered Fake News Detection System (VeritasAI)

A high-precision, explainable Machine Learning and Natural Language Processing (NLP) system for detecting fake news, sensationalism, and disinformation. The platform features statistical NLP pipelines (TF-IDF + Linear Classifiers), deep learning sequence models, token-level explainability, domain credibility scoring, and live news wire corroboration.

---

## 📋 Table of Contents

- [Features](#-features)
- [System Requirements](#-system-requirements)
- [Quick Start Guide](#-quick-start-guide)
  - [1. Open Terminal & Navigate to Project](#1-open-terminal--navigate-to-project)
  - [2. Create and Activate Virtual Environment](#2-create-and-activate-virtual-environment)
  - [3. Install Dependencies](#3-install-dependencies)
- [How to Run the Program Locally on Terminal](#-how-to-run-the-program-locally-on-terminal)
  - [Option A: Instant Terminal CLI Analysis (No Browser Needed)](#option-a-instant-terminal-cli-analysis-no-browser-needed)
  - [Option B: Run the Local Web Server & API](#option-b-run-the-local-web-server--api)
  - [Option C: Test API Endpoints via Terminal (`curl` / PowerShell)](#option-c-test-api-endpoints-via-terminal-curl--powershell)
- [Running Automated Tests](#-running-automated-tests)
- [Model Retraining (Optional)](#-model-retraining-optional)
- [Project Structure](#-project-structure)
- [Troubleshooting & FAQs](#-troubleshooting--faqs)

---

## ✨ Features

- **High-Accuracy Classification**: Pre-trained ML pipeline combining TF-IDF feature extraction with calibrated linear classifiers and deep learning models.
- **Explainable AI (XAI)**: Highlights red-flag clickbait tokens and credibility-boosting keywords with exact attribution weights.
- **Live News Wire Corroboration**: Cross-references claims against verified global news wire feeds and encyclopedic sources.
- **Multi-Modal Verification**: Analyzes headline sensationalism, stylistic risk, all-caps density, attribution markers, and domain reputation.
- **Multiple Interfaces**:
  - Direct Terminal CLI interface for rapid testing.
  - FastAPI backend with OpenAPI / Swagger documentation.
  - Integrated local HTML5 web dashboard.

---

## 💻 System Requirements

- **Python**: 3.9, 3.10, 3.11, 3.12, or 3.13
- **Operating System**: Windows 10/11, macOS, or Linux
- **Terminal Shell**: PowerShell, Command Prompt (cmd), or Bash / Zsh

---

## 🚀 Quick Start Guide

### 1. Open Terminal & Navigate to Project

Open your favorite terminal (PowerShell, Command Prompt, or Terminal) and change to the project directory:

```bash
cd "c:/AI Powered Fake News Detection"
```

### 2. Create and Activate Virtual Environment

It is recommended to use an isolated Python virtual environment.

#### On Windows (PowerShell):
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```
> *Note for PowerShell users*: If you see an execution policy error, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` and then run the activation script again.

#### On Windows (Command Prompt - `cmd`):
```cmd
python -m venv venv
venv\Scripts\activate.bat
```

#### On macOS / Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

Ensure `pip` is up to date, then install all project requirements:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🖥️ How to Run the Program Locally on Terminal

You can run the program in **three different ways** depending on your needs.

---

### Option A: Instant Terminal CLI Analysis (No Browser Needed)

Evaluate articles directly inside your terminal using `test_sample.py`.

#### 1. Run the Built-in Demo Suite (4 Realistic News Cases)
```bash
python test_sample.py --demo
```
This tests real mainstream news, viral sensational clickbait, science dispatches, and medical disinformation, printing confidence scores, red-flag keywords, and AI rationales directly to your terminal.

#### 2. Analyze Any Custom News Headline and Text
```bash
python test_sample.py --title "BREAKING: Scientists discover water on Mars" --text "NASA researchers confirmed today that satellite spectroscopy revealed subsurface ice formations in Martian craters."
```

#### CLI Options:
| Flag | Description |
| :--- | :--- |
| `--demo` | Runs the full benchmark test suite across preset samples |
| `--title "..."` | News headline or title string |
| `--text "..."` | Body content or article paragraphs |

---

### Option B: Run the Local Frontend & Web Dashboard

The frontend can be run in two ways:

#### Way 1: All-in-One Integrated Server (Recommended)
This runs the FastAPI backend and automatically serves the full frontend web dashboard on a single port:

```bash
python run_pipeline.py --mode serve --host 127.0.0.1 --port 8000
```
*(Or alternatively: `python -m src.serving.app`)*

Once running:
- Open your browser to **[http://127.0.0.1:8000](http://127.0.0.1:8000)** to use the frontend!
- API Swagger documentation is available at **[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)**.

#### Way 2: Standalone Frontend Server (Optional)
If you prefer to serve the static frontend files separately (for example, using Python's built-in HTTP server, VS Code Live Server, or Node `serve`):

1. **Terminal 1** (Start backend):
   ```bash
   python run_pipeline.py --mode serve --port 8000
   ```

2. **Terminal 2** (Start standalone frontend server):
   ```bash
   cd frontend
   python -m http.server 3000
   ```
   Then open **[http://localhost:3000](http://localhost:3000)**. The frontend will automatically detect and communicate with your backend running on port 8000.

#### Way 3: Open `index.html` Directly in Browser
You can even open [frontend/index.html](file:///c:/AI%20Powered%20Fake%20News%20Detection/frontend/index.html) directly by double-clicking it or opening it with Chrome / Edge / Firefox while the backend is running on port 8000.

---

### Option C: Test API Endpoints via Terminal (`curl` / PowerShell)

With the server running (from Option B), open a second terminal window to query the endpoints:

#### 1. Health Check
```bash
curl http://127.0.0.1:8000/health
```
**Response:**
```json
{"status":"healthy","service":"VeritasAI Enterprise Veracity Platform","version":"2.0.0"}
```

#### 2. Quick Prediction Endpoint (`/predict`)
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d "{\"title\": \"Secret cure suppressed by doctors\", \"text\": \"Miracle remedy heals all conditions in 24 hours!\"}"
```

#### 3. Full Explainability & Saliency Endpoint (`/explain`)
```bash
curl -X POST "http://127.0.0.1:8000/explain" \
  -H "Content-Type: application/json" \
  -d "{\"title\": \"Federal Reserve holds interest rates steady\", \"text\": \"Central bank officials voted unanimously to maintain rates after positive economic data.\"}"
```

#### 4. Direct Web URL Extraction & Analysis (`/api/v1/analyze-url`)
```bash
curl -X POST "http://127.0.0.1:8000/api/v1/analyze-url" \
  -H "Content-Type: application/json" \
  -d "{\"url\": \"https://en.wikipedia.org/wiki/Artificial_intelligence\"}"
```

*(Windows PowerShell alternative for curl)*:
```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/health" -Method Get
```

---

## 🧪 Running Automated Tests

Run the full automated test suite to ensure preprocessors, models, pipelines, and endpoints are operating properly:

```bash
python run_pipeline.py --mode test
```

Or using `pytest`:
```bash
pytest tests/ -v
```

Expected output:
```
test_empty_request_validation ... ok
test_health_check ... ok
test_cnn_bilstm_forward ... ok
test_evaluator_metrics ... ok
test_vocabulary_and_lstm ... ok
test_extract_stylistic_features ... ok
test_fuse_title_body ... ok
test_sanitize_wire_leakage ... ok
test_text_preprocessor_dataframe ... ok

----------------------------------------------------------------------
Ran 9 tests in 0.4s
OK
```

---

## 🔄 Model Retraining (Optional)

Trained models (`best_model.joblib`) are already included in the `artifacts/` folder. If you wish to retrain or experiment with new datasets:

```bash
# Run the complete model training pipeline
python run_pipeline.py --mode train

# Or run robust model retraining directly
python retrain_robust_model.py
```

---

## 📂 Project Structure

```
AI Powered Fake News Detection/
├── artifacts/                  # Serialized ML models and evaluation metrics
│   ├── best_model.joblib       # Active production classifier
│   └── benchmark_metrics.json  # Model accuracy and benchmark scores
├── frontend/                   # Web frontend assets (HTML, CSS, JS)
│   ├── index.html              # Main application UI
│   ├── style.css               # Styling and responsive design
│   └── app.js                  # Frontend client logic & charts
├── src/                        # Core application source code
│   ├── config.py               # Paths, hyperparameters, and environment settings
│   ├── credibility/            # Domain reputation and publisher trust registry
│   ├── data/                   # Data cleaning, text fusing, and URL scrapers
│   ├── explainability/         # TF-IDF token saliency & claim segmentation
│   ├── llm_reasoner/           # Fact-checking agent & news wire corroboration
│   ├── models/                 # Model architectures (Sklearn, BiLSTM, Ensembles)
│   └── serving/                # FastAPI backend & web server launcher
│       ├── api.py              # REST API definitions and endpoints
│       └── app.py              # Local dashboard runner and static file mount
├── tests/                      # Automated unit and integration tests
├── requirements.txt            # Python dependencies
├── run_pipeline.py             # CLI master entry point (train / serve / test)
├── test_sample.py              # Terminal CLI testing script
└── README.md                   # Project documentation
```

---

## ❓ Troubleshooting & FAQs

#### Q: `Activate.ps1 cannot be loaded because running scripts is disabled on this system` (PowerShell)
**Fix:** Run the following command in PowerShell before activating:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

#### Q: `Address already in use` error when running the server
**Fix:** Another process is already using port `8000`. You can specify a different port:
```bash
python run_pipeline.py --mode serve --port 8080
```
Then visit `http://127.0.0.1:8080`.

#### Q: `ModuleNotFoundError: No module named 'src'`
**Fix:** Ensure you are running commands from the **root directory** (`c:\AI Powered Fake News Detection`). If running modular scripts directly, set the `PYTHONPATH`:
- **PowerShell**: `$env:PYTHONPATH="."`
- **CMD**: `set PYTHONPATH=.`
- **Linux/macOS**: `export PYTHONPATH=.`

---

## 📄 License & Credits

Developed with modern Python NLP tooling (FastAPI, Scikit-Learn, PyTorch, BeautifulSoup4). Designed for research, enterprise veracity verification, and journalistic fact-checking.

//backend server
python run_pipeline.py --mode serve --host 127.0.0.1 --port 8000

//Frontend
cd frontend
 python -m http.server 3000
