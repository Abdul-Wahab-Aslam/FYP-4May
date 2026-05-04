# SecureV2V-XAI — Bahria University FYP

**Secure V2V Communication using AI** — Abdul Wahab Aslam

An Explainable AI (XAI) framework for detecting vehicular network attacks in real-time V2V communications.

---

## 🚀 Deploy on Vercel (Frontend)

### Option 1: Vercel CLI
```bash
cd frontend
npm install
npm run build
vercel --prod
```

### Option 2: Vercel GitHub Integration (Recommended)
1. Push this repo to GitHub
2. Go to [vercel.com](https://vercel.com) → New Project → Import GitHub repo
3. Set **Root Directory** → `frontend`
4. Set **Framework Preset** → `Create React App`
5. Add environment variables (optional):
   - `REACT_APP_BACKEND_URL` = your backend URL
   - `REACT_APP_WS_URL` = your WebSocket URL
   - `REACT_APP_SUPABASE_URL` = your Supabase project URL
   - `REACT_APP_SUPABASE_ANON` = your Supabase anon key
6. Click **Deploy**

---

## 🛠 Local Development

```bash
cd frontend
npm install
npm start
```
Open [http://localhost:3000](http://localhost:3000)

---

## 📁 Project Structure

```
SecureV2V-XAI/
├── frontend/               ← React app (deploy to Vercel)
│   ├── public/
│   │   ├── index.html
│   │   └── demo_accident.mp4
│   ├── src/
│   │   ├── App.js          ← Main dashboard
│   │   ├── App.css         ← Dark/light theme styles
│   │   └── bu-logo.png
│   ├── package.json
│   └── .env                ← Backend URL config
├── Backend/                ← Python FastAPI backend
│   ├── main.py
│   ├── requirements.txt
│   └── *.pkl / *.pt        ← Trained models
└── vercel.json             ← Vercel deployment config
```

---

## 🔧 Fixes Applied (Debugged Version)

| # | Issue | Fix |
|---|-------|-----|
| 1 | `@gradio/client` & `lucide-react` listed in `package.json` but not imported | Removed unused dependencies |
| 2 | `react@^19` incompatible with `react-scripts@5.0.1` | Downgraded to `react@^18.3.1` |
| 3 | Bare `catch {}` blocks (ES2019) may fail in some CRA babel configs | Changed to `catch (_err) {}` |
| 4 | Wrong WS_URL in `.env` (`/ws/v2v-metrics` instead of `/ws`) | Fixed endpoint path |
| 5 | Generic `<title>React App</title>` in `index.html` | Replaced with branded title |
| 6 | Missing `vercel.json` — SPA routes 404 on refresh | Added `vercel.json` with SPA fallback routing |
| 7 | Stale `package-lock.json` | Removed — regenerates on `npm install` |
| 8 | `__pycache__` committed to repo | Removed |

---

## 🌐 Features

- **Live V2V Radar** — animated network topology with threat detection
- **CARLA Simulation** — real-time canvas-based vehicle animation
- **YOLOv8 Dashcam Inference** — upload frames or snapshot from demo video → HF Space AI
- **SHAP Explainability** — live feature contribution chart per threat
- **Real-time Intel Feed** — WebSocket + mock fallback telemetry stream
- **Dataset KPI Bar** — accuracy/precision/recall/F1 per dataset
- **Dark/Light Theme** — full CSS variable driven theming

---

## 🧠 Models

| Model | Purpose |
|-------|---------|
| `v2v_xgboost_model.pkl` | V2V attack classification |
| `v2v_isolation_forest.pkl` | Anomaly detection |
| `v2v_scaler.pkl` | Feature normalization |
| `v2v_label_encoder.pkl` | Label encoding |
| `yolov8n.pt` | Dashcam object/crash detection |
