# V2V Sentinel XAI — Backend

FastAPI server that loads the hybrid XGBoost + LSTM models trained in
`train_veremi.py` and streams real-time threat detection over WebSocket.

## Folder Layout

```
v2v-backend/
├── main.py                 # FastAPI app (REST + WebSocket)
├── train_veremi.py         # Colab training script (run on Colab GPU)
├── requirements.txt        # Pinned dependencies
└── artifacts/              # <-- put trained .pkl files here
    ├── xgb_model.pkl
    ├── lstm_model.h5
    ├── scaler.pkl
    ├── label_encoder.pkl
    ├── shap_explainer.pkl
    └── feature_columns.json
```

## Setup (5 min)

```bash
# 1. Create a virtualenv
python -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate

# 2. Install deps
pip install -r requirements.txt

# 3. (Optional) Drop your trained artifacts/ folder in
#    The server runs in MOCK MODE if artifacts are missing — useful for
#    frontend development before the Colab job finishes.

# 4. Run
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Open http://localhost:8000/docs for the auto-generated Swagger UI.

## Endpoints

| Method | Path           | Purpose                                          |
| ------ | -------------- | ------------------------------------------------ |
| GET    | `/`            | Liveness check                                   |
| GET    | `/api/health`  | Detailed model status (loaded?, accuracy, …)     |
| GET    | `/api/metrics` | Training metrics + active vehicle/attacker count |
| POST   | `/api/predict` | One-shot threat classification + SHAP            |
| WS     | `/ws`          | Live telemetry + threat stream (~800ms cadence)  |

## WebSocket Message Format

The frontend receives one JSON object per tick:

```jsonc
{
  "type": "tick",
  "timestamp": "2026-04-26T14:32:08.234Z",
  "vehicle_id": "EV-03",
  "telemetry": {
    "sender": 3, "pos_x": 1240.3, "pos_y": 880.1, "pos_z": 0.2,
    "spd_x": 14.2, "spd_y": -3.1, "spd_z": 0.0, "...": "..."
  },
  "prediction": {
    "label": "AltSpoof",
    "confidence": 97.4,
    "probabilities": { "Normal": 1.2, "GPSSpoof": 0.8, "AltSpoof": 97.4, "SpeedInj": 0.6 },
    "explanation": [
      { "feature": "altitude_jump",    "impact": 0.42, "value": 18.3, "direction": "increases_threat" },
      { "feature": "altitude_abs",     "impact": 0.38, "value": 47.5, "direction": "increases_threat" },
      { "feature": "pos_z",            "impact": 0.31, "value": 47.5, "direction": "increases_threat" }
    ],
    "latency_ms": 24.3
  }
}
```

## Wiring it into your React frontend

Drop this into `App.js` to replace the mock `IntelFeed` data. Just three lines
of plumbing — the rest is the same component you already have:

```jsx
useEffect(() => {
  const ws = new WebSocket("ws://localhost:8000/ws");

  ws.onmessage = (e) => {
    const data = JSON.parse(e.data);
    if (data.type !== "tick") return;

    const { vehicle_id, prediction } = data;
    const isThreat = prediction.label !== "Normal";
    const msg = isThreat
      ? `[${vehicle_id}] ${prediction.label} — ${prediction.explanation[0]?.feature ?? ""}`
      : `[${vehicle_id}] Normal BSM — Clear`;

    setLogs(prev => [{
      ts:   new Date().toTimeString().slice(0, 8) + "." + String(Math.floor(Math.random()*999)).padStart(3,"0"),
      msg,
      conf: prediction.confidence,
    }, ...prev].slice(0, 20));
  };

  return () => ws.close();
}, []);
```

For production, swap `ws://localhost:8000/ws` for your deployed URL
(e.g., `wss://v2v-api.onrender.com/ws`) via an env var:
`process.env.REACT_APP_WS_URL`.

## Deployment

The backend deploys cleanly to:
- **Render** (free tier, supports WebSocket)
- **Railway** (one-click Python deploy)
- **Fly.io** (best for low-latency WS)

⚠️ Vercel does **not** support persistent WebSockets on its serverless tier —
keep the React frontend on Vercel and the FastAPI backend on Render/Railway.

### Slim deployment without TensorFlow

If you only ship the XGBoost model (which alone hits ~96–98% on VeReMi),
comment out the `tensorflow` line in `requirements.txt` — your Docker image
drops from ~600 MB to ~80 MB. The code already gracefully falls back to
XGB-only inference if the LSTM file is missing.

## Mock Mode

If `./artifacts/` is missing or incomplete, the server boots in **MOCK MODE**:
predictions are realistic random data with the same schema as real ones. This
lets you build/test the React side while the Colab training is still running.
Look for `"_mock": true` in the prediction payload to detect it.
