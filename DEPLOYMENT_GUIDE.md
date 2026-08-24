# End-to-End Deployment Guide: Netlify (Frontend) + Render / Railway (Backend)

This guide walks you through deploying your **AI-Powered Fake News Detection System** to the cloud using **Netlify** (Frontend) and **Render** or **Railway** (Backend API).

---

## 📁 Repository Structure Overview

```
c:/AI Powered Fake News Detection/
├── frontend/                     # Ready for Netlify Deployment
│   ├── index.html                # UI Layout & Preset Buttons
│   ├── style.css                 # Responsive Design & Theme
│   ├── app.js                    # API Client & Real-Time Charts
│   ├── netlify.toml              # Netlify Build & CORS Config
│   └── _redirects                # Routing Rules
├── src/                          # FastAPI Backend Source Code
│   ├── serving/
│   │   ├── api.py                # REST API with CORS enabled
│   │   └── app.py                # Local Server Runner
│   └── ...
├── artifacts/                    # Trained Production Models (97.49% Acc)
│   ├── best_model.joblib         # Serialized ML Pipeline
│   └── ...
├── requirements.txt              # Production Python Dependencies
├── Procfile                      # Web process launcher (Render/Railway)
├── render.yaml                   # 1-Click Render Blueprint
└── Dockerfile                    # Container definition
```

---

## 🚀 STEP 1: Deploy Backend to Render (Free Web Service)

[Render.com](https://render.com) provides free cloud hosting for FastAPI applications.

### Option A: Via GitHub (Recommended)
1. Push this project repository to **GitHub**.
2. Go to [https://dashboard.render.com](https://dashboard.render.com) and click **New +** -> **Web Service**.
3. Connect your GitHub repository.
4. Fill in the service configuration:
   - **Name**: `fake-news-detector-api`
   - **Environment**: `Python`
   - **Region**: Select closest to you (e.g., Oregon / Frankfurt / Singapore)
   - **Branch**: `main`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn src.serving.api:app --host 0.0.0.0 --port $PORT`
   - **Plan**: `Free`
5. Click **Create Web Service**.
6. Once deployed, Render will provide a live URL like:
   `https://fake-news-detector-api.onrender.com`
7. Test the backend by opening:
   `https://fake-news-detector-api.onrender.com/health` (should return `{"status": "healthy"}`).

---

## 🌐 STEP 2: Deploy Frontend to Netlify

[Netlify](https://www.netlify.com) provides free global hosting for modern web frontends.

### Option A: Drag & Drop (Instant - 30 Seconds)
1. Go to [https://app.netlify.com/drop](https://app.netlify.com/drop).
2. Drag and drop the **`frontend`** folder directly into the browser window.
3. Netlify will publish your site immediately and generate a live URL (e.g. `https://veritas-fakenews.netlify.app`).

### Option B: Via GitHub Repository
1. In the [Netlify Dashboard](https://app.netlify.com), click **Add new site** -> **Import an existing project**.
2. Choose **GitHub** and select your repository.
3. In the build settings:
   - **Base directory**: `frontend`
   - **Publish directory**: `.` (or leave as `frontend`)
   - **Build command**: *(leave blank)*
4. Click **Deploy site**.

---

## 🔗 STEP 3: Connect Frontend to Your Hosted Backend

1. Open your published Netlify URL (e.g., `https://veritas-fakenews.netlify.app`).
2. In the top-right header, find the **Backend URL** input field.
3. Paste your Render backend URL (e.g., `https://fake-news-detector-api.onrender.com`).
4. Click the **✓ Checkmark** button to save.
5. Click any preset sample or paste custom news text and click **Run AI Classification**!

---

## 💡 Alternative Backend Hosting: Railway.app

If you prefer **Railway**:
1. Go to [https://railway.app](https://railway.app) and click **New Project** -> **Deploy from GitHub repo**.
2. Select your repository.
3. Railway automatically detects the `Procfile` and `requirements.txt`.
4. Under Settings -> Networking, click **Generate Domain** to get your public API URL.
