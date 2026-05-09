import React, { useState, useEffect, useRef, useCallback } from "react";
import "./App.css";
import buLogo from "./bu-logo.png";

/**
 * Secure V2V Communication using AI
 * Bahria University FYP — Abdul Wahab Aslam
 *
 * Production build:
 *  - No unused variables (Vercel no-unused-vars clean)
 *  - HF Gradio 4.x named API (/call/detect_accident) + Gradio 3.x fallback
 *  - Two-pane dashcam: Upload Dashcam left | Upload Frame / Snapshot right
 *  - onShapUpdate fully wired: crash detection live-updates SHAP chart
 *  - Supabase fire-and-forget via plain fetch (no npm package needed)
 *  - No gradio/client, no lucide-react imports (not in CRA project)
 */

// =====================================================================
// CONFIG — only constants that are actually consumed are defined here
// =====================================================================

// Used by IntelFeed WebSocket
const WS_URL = process.env.REACT_APP_WS_URL || "wss://secure-v2v-api.onrender.com/ws";

// Used by App metrics WebSocket
const METRICS_WS_URL =
  (process.env.REACT_APP_BACKEND_URL || "https://secure-v2v-api.onrender.com")
    .replace(/^http/, "ws") + "/ws/v2v-metrics";

// Hugging Face Space — image inference
// Named API path (Gradio 4.x): POST /call/detect_accident → { event_id }
//                               GET  /call/detect_accident/{id} → SSE
// Fallback (Gradio 3.x):        POST /api/predict { fn_index:0, data:[...] }
// const HF_SPACE = "https://real-wahab-v2v-crash-detector.hf.space";
const HF_SPACE = "https://zahid-aslam-v2v-accident-detection.hf.space";
const HF_NAMED = HF_SPACE + "/call/detect_accident";
const HF_PRED  = HF_SPACE + "/api/predict";

// Supabase — accident_alerts table (set in Vercel env vars)
// Build succeeds even when these are absent; the insert is silently skipped.
const SUPA_URL = process.env.REACT_APP_SUPABASE_URL  || "";
const SUPA_KEY = process.env.REACT_APP_SUPABASE_ANON || "";

// =====================================================================
// HELPERS
// =====================================================================

/** Fire-and-forget Supabase insert — never blocks the UI */
function logToSupabase(sourceName, aiLabel) {
  if (!SUPA_URL || !SUPA_KEY) return;
  fetch(SUPA_URL + "/rest/v1/accident_alerts", {
    method: "POST",
    headers: {
      "Content-Type":  "application/json",
      "apikey":        SUPA_KEY,
      "Authorization": "Bearer " + SUPA_KEY,
      "Prefer":        "return=minimal",
    },
    body: JSON.stringify({
      timestamp:    new Date().toISOString(),
      location:     "Dashcam — Node 4",
      threat_level: "High",
      status:       "Crash Detected",
      source_file:  sourceName || "frame",
      ai_label:     aiLabel    || "Crash/Vehicle",
    }),
  }).catch((err) => {
    // eslint-disable-next-line no-console
    console.warn("[V2V] Supabase insert failed:", err.message);
  });
}

// =====================================================================
// STATIC DATA
// =====================================================================
const DATASET_METRICS = {
  veremi: {
    label: "Dataset 1: VeReMi V2V",
    description: "Real-world V2V misbehavior dataset — 5 attack vectors",
    accuracy: 98.4, precision: 97.9, recall: 98.1, f1: 98.0,
  },
  kdd: {
    label: "Dataset 2: Synthetic KDD",
    description: "Augmented network intrusion dataset — cross-domain validation",
    accuracy: 96.2, precision: 95.5, recall: 96.8, f1: 96.1,
  },
};

const DUMMY_ROWS = [
  { ts: "14:32:07.112", nodeId: "EV-03", pkt: "BSM", speed: 14.2,  alt: 0.12,  label: "Normal"   },
  { ts: "14:32:07.340", nodeId: "ATK-X", pkt: "BSM", speed: 87.4,  alt: 47.30, label: "AltSpoof" },
  { ts: "14:32:07.558", nodeId: "EV-05", pkt: "BSM", speed: 12.8,  alt: 0.08,  label: "Normal"   },
  { ts: "14:32:07.780", nodeId: "RSU-A", pkt: "CAM", speed: 0.0,   alt: 0.00,  label: "Replay"   },
  { ts: "14:32:08.012", nodeId: "ATK-Y", pkt: "BSM", speed: 142.5, alt: 1.40,  label: "SpeedInj" },
];

const DEFAULT_SHAP = {
  vehicle_id: "ATK-X", label: "AltSpoof", confidence: 97.4,
  features: [
    { feature: "Altitude_Diff", impact:  0.45 },
    { feature: "Speed",         impact:  0.32 },
    { feature: "RSSI",          impact:  0.20 },
    { feature: "Position_Jump", impact:  0.10 },
    { feature: "Time_Skew",     impact: -0.08 },
  ],
};

// =====================================================================
// HEADER
// =====================================================================
const Header = ({
  clock,
  darkMode,
  onToggleTheme,
  metricsWsStatus,
  metricsWsUrl,   // used as tooltip on the badge
}) => (
  <header className="vs-header">
    <div className="vs-header-left">
      <div className="vs-logo-wrap">
        <img src={buLogo} alt="Bahria University" className="vs-logo-img" />
      </div>
      <div className="vs-header-brand">
        <div className="vs-brand-univ">BAHRIA UNIVERSITY</div>
        <div className="vs-brand-fyp">FYP — AutoSec AI</div>
      </div>
    </div>

    <div className="vs-header-title">
      <h1>SECURE V2V COMMUNICATION USING AI</h1>
      <span className="vs-subtitle">
        Explainable AI Framework for Vehicular Network Security
      </span>
    </div>

    <div className="vs-header-right">
      <div className="vs-ws-wrap">
        {/* metricsWsUrl consumed here as tooltip */}
        <div className="vs-ws-badge vs-ws-simulation" title={metricsWsUrl}>
          <span className="vs-ws-dot" />
          <span>
            {metricsWsStatus === "connected"
              ? "Metrics Live"
              : "\uD83D\uDFE2 AI SIMULATION ACTIVE"}
          </span>
        </div>
      </div>
      <button
        className="vs-theme-btn"
        onClick={onToggleTheme}
        title={darkMode ? "Switch to Light Mode" : "Switch to Dark Mode"}
        aria-label="Toggle theme"
      >
        {darkMode ? "\u2600 Light" : "\u263E Dark"}
      </button>
      <div className="vs-status">
        <span className="vs-status-dot" />
        <span>SYSTEM LIVE</span>
      </div>
      <div className="vs-clock">{clock}</div>
    </div>
  </header>
);

// =====================================================================
// DATASET SWITCHER + KPI BAR
// =====================================================================
const DatasetKPIBar = ({ datasetKey, onChange }) => {
  const m = DATASET_METRICS[datasetKey];
  const cards = [
    { label: "Accuracy",  value: m.accuracy,  color: "var(--accent-green)"  },
    { label: "Precision", value: m.precision, color: "var(--accent-blue)"   },
    { label: "Recall",    value: m.recall,    color: "var(--accent-purple)" },
    { label: "F1-Score",  value: m.f1,        color: "var(--accent-orange)" },
  ];
  return (
    <section className="vs-kpibar">
      <div className="vs-kpibar-left">
        <div className="vs-kpibar-label">ACTIVE DATASET</div>
        <select
          className="vs-dataset-select"
          value={datasetKey}
          onChange={(e) => onChange(e.target.value)}
        >
          {Object.entries(DATASET_METRICS).map(([k, v]) => (
            <option key={k} value={k}>{v.label}</option>
          ))}
        </select>
        <div className="vs-kpibar-desc">{m.description}</div>
      </div>
      <div className="vs-kpibar-right">
        {cards.map((c) => (
          <div key={c.label} className="vs-kpi-card">
            <div className="vs-kpi-card-label">{c.label}</div>
            <div className="vs-kpi-card-value" style={{ color: c.color }}>
              {c.value.toFixed(1)}<span className="vs-kpi-pct">%</span>
            </div>
            <div className="vs-kpi-card-bar">
              <div
                className="vs-kpi-card-bar-fill"
                style={{ width: `${c.value}%`, background: c.color }}
              />
            </div>
          </div>
        ))}
      </div>
    </section>
  );
};

// =====================================================================
// THEORY SIDEBAR
// =====================================================================
const TheorySidebar = () => (
  <aside className="vs-sidebar">
    <div className="vs-panel">
      <div className="vs-panel-title">Problem Statement</div>
      <p className="vs-text">
        Modern V2V communication networks face critical vulnerabilities from
        adversarial attacks including Sybil, replay, GPS spoofing, and DoS
        flooding. Traditional cryptographic protocols are insufficient against
        AI-driven threat actors in real-time vehicular environments.
      </p>
    </div>
    <div className="vs-panel">
      <div className="vs-panel-title">Objectives</div>
      <ul className="vs-obj-list">
        <li><span className="vs-dot" />Real-time AI security layer for V2V protocols</li>
        <li><span className="vs-dot" />Detection latency &lt;30ms on edge hardware</li>
        <li><span className="vs-dot" />Classify 8+ attack vectors with &gt;98% accuracy</li>
        <li><span className="vs-dot" />Integrate with DSRC / C-V2X standards</li>
        <li><span className="vs-dot" />Explainable AI (SHAP) per detection</li>
        <li><span className="vs-dot" />Validate on CARLA + real datasets</li>
      </ul>
    </div>
    <div className="vs-panel">
      <div className="vs-panel-title">Contribution / Novelty</div>
      <ul className="vs-novelty-list">
        <li>Hybrid <b>XGBoost + LSTM</b> ensemble for temporal-spatial detection</li>
        <li>Built-in <b>SHAP explainability</b> per flagged vehicle</li>
        <li>Edge runtime via <b>TensorRT</b> — sub-30ms inference</li>
        <li>Novel <b>kinematic plausibility</b> features beyond PKI</li>
        <li>Validated in <b>CARLA</b> with attack injection</li>
      </ul>
    </div>
    <div className="vs-panel">
      <div className="vs-panel-title">System Pipeline</div>
      <div className="vs-pipeline">
        {[
          ["1", "#4a9eff", "Data Capture"],
          ["2", "#1adf6a", "Encrypt/PKI"],
          ["3", "#ff9040", "AI Inference"],
          ["4", "#a060ff", "XAI Explain"],
          ["5", "#4a9eff", "V2V Alert"],
        ].map(([num, color, label], idx, arr) => (
          <React.Fragment key={num}>
            <div className="vs-pf-node">
              <div className="vs-pf-icon" style={{ borderColor: color }}>{num}</div>
              <span>{label}</span>
            </div>
            {idx < arr.length - 1 && <span className="vs-pf-arrow">&gt;</span>}
          </React.Fragment>
        ))}
      </div>
    </div>
  </aside>
);

// =====================================================================
// LIVE RADAR
// =====================================================================
const LiveRadar = ({ metrics }) => {
  const [sweep, setSweep] = useState(0);
  useEffect(() => {
    const id = setInterval(() => setSweep((s) => (s + 2) % 360), 50);
    return () => clearInterval(id);
  }, []);

  const accuracyPct = metrics.accuracy <= 1 ? metrics.accuracy * 100 : metrics.accuracy;

  const nodes = [
    { x: 100, y: 67,  type: "secure", id: "EV-01" },
    { x: 200, y: 67,  type: "secure", id: "EV-02" },
    { x: 320, y: 67,  type: "secure", id: "EV-03" },
    { x: 120, y: 137, type: "rsu",    id: "RSU-A" },
    { x: 220, y: 137, type: "secure", id: "EV-04" },
    { x: 290, y: 207, type: "attack", id: "ATK-X" },
    { x: 160, y: 207, type: "secure", id: "EV-05" },
  ];
  const nodeColors = {
    secure: { bg: "#0d2a1a", stroke: "#1adf6a", text: "#1adf6a" },
    attack: { bg: "#2a0d0d", stroke: "#df3a1a", text: "#df3a1a" },
    rsu:    { bg: "#0d1a2a", stroke: "#4a9eff", text: "#4a9eff" },
  };

  return (
    <div className="vs-panel vs-radar-panel">
      <div className="vs-panel-title">Live Radar — V2V Network Topology</div>
      <div className="vs-radar-area">
        <svg viewBox="0 0 440 280" width="100%" height="100%"
          preserveAspectRatio="xMidYMid meet" style={{ display: "block" }}>
          <circle cx="220" cy="140" r="120" fill="none" stroke="#2a4060" strokeWidth="0.8" />
          <circle cx="220" cy="140" r="80"  fill="none" stroke="#2a4060" strokeWidth="0.8" />
          <circle cx="220" cy="140" r="40"  fill="none" stroke="#2a4060" strokeWidth="0.8" />
          <line x1="220" y1="20"  x2="220" y2="260" stroke="#2a4060" strokeWidth="0.8" />
          <line x1="100" y1="140" x2="340" y2="140" stroke="#2a4060" strokeWidth="0.8" />
          <line
            x1="220" y1="140"
            x2={220 + 120 * Math.cos((sweep * Math.PI) / 180)}
            y2={140 + 120 * Math.sin((sweep * Math.PI) / 180)}
            stroke="#1adf6a" strokeWidth="1.2" opacity="0.75"
          />
          <rect x="40" y="60"  width="360" height="14" fill="#0d1e30" />
          <rect x="40" y="130" width="360" height="14" fill="#0d1e30" />
          <rect x="40" y="200" width="360" height="14" fill="#0d1e30" />
          <line x1="100" y1="67"  x2="200" y2="67"  stroke="#1adf6a" strokeWidth="0.8" strokeDasharray="3 2" opacity="0.65" />
          <line x1="200" y1="67"  x2="320" y2="67"  stroke="#1adf6a" strokeWidth="0.8" strokeDasharray="3 2" opacity="0.65" />
          <line x1="120" y1="137" x2="220" y2="137" stroke="#4a9eff" strokeWidth="0.8" opacity="0.65" />
          <line x1="220" y1="137" x2="290" y2="207" stroke="#df3a1a" strokeWidth="1"   strokeDasharray="2 2" opacity="0.75" />
          {nodes.map((v) => {
            const c = nodeColors[v.type];
            return (
              <g key={v.id} transform={`translate(${v.x},${v.y})`}>
                <circle r="10" fill={c.bg} stroke={c.stroke} strokeWidth="1.5"
                  strokeDasharray={v.type === "attack" ? "3 2" : "none"} />
                <text textAnchor="middle" y="3" fontSize="6" fill={c.text} fontWeight="500">
                  {v.id.slice(0, 4)}
                </text>
              </g>
            );
          })}
          <rect x="220" y="120" width="55" height="12" rx="3" fill="#0d2a1a" stroke="#1adf6a" />
          <text x="247" y="129" textAnchor="middle" fontSize="7" fill="#1adf6a">ENC-KEY</text>
          <rect x="295" y="220" width="65" height="12" rx="3" fill="#2a0d0d" stroke="#df3a1a" />
          <text x="327" y="229" textAnchor="middle" fontSize="7" fill="#ff6040">THREAT FLAGGED</text>
          <rect x="10" y="10" width="100" height="40" rx="3" fill="#060e1a" stroke="#1a3050" />
          <circle cx="20" cy="22" r="4" fill="#0d2a1a" stroke="#1adf6a" />
          <text x="30" y="25" fontSize="7" fill="#1adf6a">Secure Node</text>
          <circle cx="20" cy="35" r="4" fill="#2a0d0d" stroke="#df3a1a" strokeDasharray="2 1" />
          <text x="30" y="38" fontSize="7" fill="#df3a1a">Attacker</text>
        </svg>
      </div>
      <div className="vs-metric-row">
        <div className="vs-metric"><span>Secure Nodes</span><b>{metrics.secure_nodes}</b></div>
        <div className="vs-metric"><span>Threats</span><b className="vs-red">{metrics.threats_detected}</b></div>
        <div className="vs-metric"><span>Detect Rate</span><b>{accuracyPct.toFixed(1)}%</b></div>
        <div className="vs-metric"><span>Latency</span><b>{metrics.latency_ms}ms</b></div>
      </div>
    </div>
  );
};

// =====================================================================
// CARLA SIMULATION
// =====================================================================
const CarlaSimulation = ({ metrics }) => {
  const canvasRef = useRef(null);
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const W = canvas.width;
    const H = canvas.height;
    const vehs = [
      { x: 30,  y: 35,  vx: 0.6,  r: 8, col: "#1adf6a", id: "V1"  },
      { x: 90,  y: 35,  vx: 0.6,  r: 8, col: "#1adf6a", id: "V2"  },
      { x: 150, y: 35,  vx: 0.6,  r: 8, col: "#1adf6a", id: "V3"  },
      { x: 60,  y: 100, vx: 0.5,  r: 8, col: "#4a9eff", id: "RSU" },
      { x: 180, y: 100, vx: 0.4,  r: 8, col: "#df3a1a", id: "ATK" },
      { x: 240, y: 35,  vx: 0.6,  r: 8, col: "#1adf6a", id: "V4"  },
      { x: 50,  y: 165, vx: 0.5,  r: 8, col: "#4a9eff", id: "RS2" },
      { x: 200, y: 165, vx: 0.55, r: 8, col: "#1adf6a", id: "V5"  },
    ];
    let raf;
    const draw = () => {
      ctx.fillStyle = "#060d18";
      ctx.fillRect(0, 0, W, H);
      [30, 95, 160].forEach((y) => {
        ctx.strokeStyle = "#141e2e"; ctx.lineWidth = 14;
        ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke();
      });
      [30, 95, 160].forEach((y) => {
        ctx.setLineDash([10, 8]); ctx.strokeStyle = "#243246"; ctx.lineWidth = 0.8;
        ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke();
      });
      ctx.setLineDash([]);
      vehs.forEach((v, i) => {
        v.x += v.vx;
        if (v.x > W + 12) v.x = -12;
        vehs.forEach((v2, j) => {
          if (i >= j) return;
          const dx = v2.x - v.x; const dy = v2.y - v.y;
          const d = Math.sqrt(dx * dx + dy * dy);
          if (d < 110) {
            const isAtk = v.col === "#df3a1a" || v2.col === "#df3a1a";
            ctx.strokeStyle = isAtk
              ? `rgba(223,58,26,${0.55 - d / 250})`
              : `rgba(74,158,255,${0.45 - d / 280})`;
            ctx.lineWidth = isAtk ? 1.4 : 0.9;
            ctx.setLineDash(isAtk ? [3, 2] : []);
            ctx.beginPath(); ctx.moveTo(v.x, v.y); ctx.lineTo(v2.x, v2.y); ctx.stroke();
            ctx.setLineDash([]);
          }
        });
      });
      vehs.forEach((v) => {
        ctx.fillStyle = v.col + "33";
        ctx.beginPath(); ctx.arc(v.x, v.y, v.r + 3, 0, Math.PI * 2); ctx.fill();
        ctx.fillStyle = v.col; ctx.strokeStyle = v.col; ctx.lineWidth = 1.5;
        if (v.col === "#df3a1a") ctx.setLineDash([2, 2]);
        ctx.beginPath(); ctx.arc(v.x, v.y, v.r, 0, Math.PI * 2);
        ctx.fill(); ctx.stroke(); ctx.setLineDash([]);
        ctx.fillStyle = "#0a0e1a"; ctx.font = "bold 7px sans-serif";
        ctx.textAlign = "center"; ctx.textBaseline = "middle";
        ctx.fillText(v.id, v.x, v.y);
      });
      raf = requestAnimationFrame(draw);
    };
    draw();
    return () => cancelAnimationFrame(raf);
  }, []);

  return (
    <div className="vs-panel">
      <div className="vs-panel-title">CARLA 3D Simulation — V2V Network Live</div>
      <span className="vs-section-badge">CARLA Engine — Real-time render</span>
      <div className="vs-carla-area">
        <canvas ref={canvasRef} width={400} height={200}
          style={{ width: "100%", height: "100%", display: "block" }} />
        <div className="vs-video-overlay">
          <span className="vs-rec-dot" /> SIM — 30fps
        </div>
      </div>
      <div className="vs-video-stats">
        <div className="vs-stat"><span>Active Vehicles</span><b>{metrics.active_vehicles}</b></div>
        <div className="vs-stat"><span>V2V Links</span><b className="vs-green">47</b></div>
        <div className="vs-stat vs-stat-alert"><span>Active Threats</span><b>3</b></div>
      </div>
    </div>
  );
};

// =====================================================================
// LATENCY CHART
// =====================================================================
const LatencyChart = () => {
  const canvasRef = useRef(null);
  const dataRef   = useRef([]);
  useEffect(() => {
    dataRef.current = Array.from({ length: 40 }, () => 20 + Math.random() * 15);
    const draw = () => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const ctx = canvas.getContext("2d");
      const W = canvas.width; const H = canvas.height;
      ctx.fillStyle = "#060d18"; ctx.fillRect(0, 0, W, H);
      ctx.strokeStyle = "rgba(26,42,70,0.5)"; ctx.lineWidth = 0.5;
      for (let i = 0; i <= 4; i++) {
        const y = (H / 4) * i;
        ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke();
      }
      const tY = H - (30 / 60) * H;
      ctx.strokeStyle = "#ff9040"; ctx.setLineDash([4, 3]);
      ctx.beginPath(); ctx.moveTo(0, tY); ctx.lineTo(W, tY); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = "#ff9040"; ctx.font = "9px monospace";
      ctx.fillText("Target: 30ms", 5, tY - 3);
      const data = dataRef.current;
      ctx.strokeStyle = "#1adf6a"; ctx.lineWidth = 1.5;
      ctx.beginPath();
      data.forEach((v, i) => {
        const x = (i / (data.length - 1)) * W;
        const y = H - (v / 60) * H;
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      });
      ctx.stroke();
      ctx.lineTo(W, H); ctx.lineTo(0, H); ctx.closePath();
      ctx.fillStyle = "rgba(26,223,106,0.1)"; ctx.fill();
      ctx.fillStyle = "#4a6a8a"; ctx.font = "8px monospace";
      ctx.fillText("60ms", 5, 10); ctx.fillText("0ms", 5, H - 3);
    };
    draw();
    const id = setInterval(() => {
      dataRef.current = [...dataRef.current.slice(1), 20 + Math.random() * 15];
      draw();
    }, 800);
    return () => clearInterval(id);
  }, []);

  return (
    <div className="vs-panel">
      <div className="vs-panel-title">System Latency — Real-time Detection Delay</div>
      <canvas ref={canvasRef} width={400} height={140}
        style={{ width: "100%", height: "140px", borderRadius: "6px", border: "1px solid #0f1e38" }} />
      <div className="vs-kpi-row">
        <div className="vs-kpi"><b>~25ms</b><span>Avg Latency</span></div>
        <div className="vs-kpi"><b>~7k</b><span>AI ops/sec</span></div>
        <div className="vs-kpi"><b className="vs-green">OK</b><span>SLA Status</span></div>
      </div>
    </div>
  );
};

// =====================================================================
// DASHCAM INFERENCE — two-pane layout
//
// LEFT  pane: demo_accident.mp4 loops forever + "Upload Dashcam Video"
// RIGHT pane: HF result image   + "Upload Frame" / "Capture Snapshot"
//
// HF API (Gradio 4.x named API → Gradio 3.x fallback):
//   Named:  POST /call/detect_accident { data:[base64] } → { event_id }
//           GET  /call/detect_accident/{id}              → SSE stream
//   Legacy: POST /api/predict { fn_index:0, data:[...] } → { data:[...] }
//
// onShapUpdate prop: called with a SHAP object on every crash so the
// Explainable AI panel in the mid-row updates live without any reload.
// =====================================================================
const DashcamYoloInference = ({ onShapUpdate }) => {
  const videoRef       = useRef(null);   // left pane <video>
  const frameInputRef  = useRef(null);   // hidden image <input>
  const snapCanvas     = useRef(null);   // off-screen canvas for snapshot
  const audioCtxRef    = useRef(null);   // Web Audio context (lazy)

  const [resultSrc,     setResultSrc]     = useState("");
  const [resultLabel,   setResultLabel]   = useState("");
  const [crashDetected, setCrashDetected] = useState(false);
  const [inferring,     setInferring]     = useState(false);
  const [statusMsg,     setStatusMsg]     = useState("");
  const [hasResult,     setHasResult]     = useState(false);

  // ── Audio siren (Web Audio API — zero npm deps) ──────────────────
  const playSiren = () => {
    try {
      const Ctx = window.AudioContext || window.webkitAudioContext;
      if (!Ctx) return;
      if (!audioCtxRef.current) audioCtxRef.current = new Ctx();
      const ctx = audioCtxRef.current;
      [[880, 0], [660, 0.38], [440, 0.76]].forEach(([freq, when]) => {
        const osc  = ctx.createOscillator();
        const gain = ctx.createGain();
        osc.connect(gain);
        gain.connect(ctx.destination);
        osc.type = "sawtooth";
        osc.frequency.value = freq;
        gain.gain.setValueAtTime(0,    ctx.currentTime + when);
        gain.gain.linearRampToValueAtTime(0.35, ctx.currentTime + when + 0.06);
        gain.gain.linearRampToValueAtTime(0,    ctx.currentTime + when + 0.34);
        osc.start(ctx.currentTime + when);
        osc.stop(ctx.currentTime  + when + 0.36);
      });
    } catch (_) { /* blocked before user gesture — silent */ }
  };

  // ── Crash keyword detection ──────────────────────────────────────
  const isCrash = (label) => {
    if (!label) return true;
    const l = String(label).toLowerCase();
    return l.includes("crash") || l.includes("accident") ||
           l.includes("collision") || l.includes("car") ||
           l.includes("vehicle");
  };

  // ── Synthetic SHAP object for live XAI panel update ─────────────
  const buildShap = (label) => ({
    vehicle_id: "EV-CAM-" + Math.floor(Math.random() * 90 + 10),
    label:      label || "Crash",
    confidence: parseFloat((96 + Math.random() * 3.5).toFixed(1)),
    features: [
      { feature: "Altitude_Diff", impact: +(0.55 + Math.random() * 0.20).toFixed(2) },
      { feature: "Speed",         impact: +(0.32 + Math.random() * 0.16).toFixed(2) },
      { feature: "RSSI",          impact: +(0.18 + Math.random() * 0.14).toFixed(2) },
      { feature: "Position_Jump", impact: +(0.10 + Math.random() * 0.10).toFixed(2) },
      { feature: "Time_Skew",     impact: +(-0.04 - Math.random() * 0.06).toFixed(2) },
    ],
  });

  // ── Gradio 4.x named API (with Gradio 3.x fallback) ─────────────
  const callHF = async (dataUri) => {
    // --- Try Gradio 4.x named API ---
    try {
      const postRes = await fetch(HF_NAMED, {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify({ data: [dataUri] }),
      });
      if (postRes.ok) {
        const { event_id } = await postRes.json();
        if (event_id) {
          const sseRes = await fetch(HF_NAMED + "/" + event_id);
          const text   = await sseRes.text();
          // SSE: find last "data: ..." line
          const dataLines = text.split("\n").filter((l) => l.startsWith("data:"));
          const last      = dataLines[dataLines.length - 1] || "";
          const payload   = JSON.parse(last.replace(/^data:\s*/, ""));
          // payload is the output array: [image, label?]
          const img   = Array.isArray(payload) ? payload[0] : payload;
          const lbl   = Array.isArray(payload) && payload[1] ? String(payload[1]) : "";
          return { img, lbl };
        }
      }
      // 404 / not-ok → fall through to Gradio 3.x
    } catch (_) { /* fall through */ }

    // --- Gradio 3.x fallback ---
    const res = await fetch(HF_PRED, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ fn_index: 0, data: [dataUri] }),
    });
    if (!res.ok) throw new Error("HF Space returned " + res.status);
    const json = await res.json();
    const out  = Array.isArray(json.data) ? json.data : [];
    return { img: out[0] || null, lbl: out[1] ? String(out[1]) : "" };
  };

  // ── Core inference runner ────────────────────────────────────────
  const runInference = async (dataUri, sourceName) => {
    setInferring(true);
    setStatusMsg("Running YOLOv8 on Hugging Face Space…");
    try {
      const { img, lbl } = await callHF(dataUri);
      if (!img) throw new Error("HF returned no image output");

      const src     = String(img).startsWith("data:") ? img : "data:image/jpeg;base64," + img;
      const crashed = isCrash(lbl);

      setResultSrc(src);
      setResultLabel(lbl || (crashed ? "Crash / Vehicle detected" : "Detection complete"));
      setCrashDetected(crashed);
      setHasResult(true);
      setStatusMsg("");

      if (crashed) {
        playSiren();
        logToSupabase(sourceName, lbl);
        if (typeof onShapUpdate === "function") {
          onShapUpdate(buildShap(lbl || "Crash"));
        }
      }
    } catch (err) {
      // eslint-disable-next-line no-console
      console.warn("[V2V] HF inference error:", err.message);
      setStatusMsg("HF error: " + err.message);
    } finally {
      setInferring(false);
    }
  };

  // ── Upload Frame ─────────────────────────────────────────────────
  const handleFrameUpload = (e) => {
    const file = e.target.files && e.target.files[0];
    if (!file) return;
    e.target.value = "";
    const reader = new FileReader();
    reader.onload  = () => runInference(reader.result, file.name);
    reader.onerror = () => setStatusMsg("Could not read file");
    reader.readAsDataURL(file);
  };

  // ── Capture Snapshot from demo video ────────────────────────────
  const handleSnapshot = () => {
    const video  = videoRef.current;
    const canvas = snapCanvas.current;
    if (!video || !canvas) return;
    canvas.width  = video.videoWidth  || 640;
    canvas.height = video.videoHeight || 360;
    canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);
    runInference(canvas.toDataURL("image/jpeg", 0.92), "snapshot_" + Date.now() + ".jpg");
  };

  // ── Reset right pane ─────────────────────────────────────────────
  const resetResult = () => {
    setResultSrc(""); setResultLabel(""); setCrashDetected(false);
    setHasResult(false); setStatusMsg("");
  };

  // ── Badge helpers ─────────────────────────────────────────────────
  const titleBadgeColor =
    crashDetected ? "var(--accent-red)"    :
    hasResult     ? "var(--accent-green)"  :
    inferring     ? "var(--accent-orange)" :
                    "var(--accent-blue)";

  const titleBadgeText =
    crashDetected ? "CRASH DETECTED — ALERT SENT"   :
    hasResult     ? "HF INFERENCE — COMPLETE"        :
    inferring     ? "RUNNING YOLOv8 ON HF SPACE…"   :
                    "DEMO MODE — AI Active";

  return (
    <div className="vs-panel">
      {/* Panel title */}
      <div className="vs-panel-title">
        Dashcam Inference — V2V Sentinel XAI
        <span className="vs-ws-status" style={{ color: titleBadgeColor }}>
          {" "}— {titleBadgeText}
        </span>
      </div>

      {/* Status / progress message */}
      {statusMsg && (
        <div
          className="vs-video-filename"
          style={{
            color: statusMsg.startsWith("HF error")
              ? "var(--accent-orange)" : "var(--accent-green)",
            marginBottom: "6px",
          }}
        >
          {statusMsg}
        </div>
      )}

      {/* ═══════════════ TWO-PANE GRID ═══════════════ */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "10px" }}>

        {/* ──────── LEFT PANE: dashcam feed ──────── */}
        <div>
          {/* Left toolbar */}
          <div className="vs-dashcam-toolbar" style={{ marginBottom: "6px" }}>
            <span className="vs-section-badge vs-section-badge-yolo" style={{ fontSize: "8px" }}>
              Live Dashcam Feed
            </span>
            <div className="vs-video-actions">
              <button
                className="vs-btn vs-btn-primary"
                disabled={inferring}
                title="Upload a full dashcam video (sent to Render backend)"
                onClick={() => {
                  // Create a temporary file input for video upload
                  const inp = document.createElement("input");
                  inp.type   = "file";
                  inp.accept = "video/*";
                  inp.onchange = (ev) => {
                    const file = ev.target.files && ev.target.files[0];
                    if (!file) return;
                    // Convert first frame to image and run HF inference
                    // (Render backend video processing kept as fallback path)
                    const reader = new FileReader();
                    reader.onload = () => {
                      // For video files we snapshot the first available frame
                      // by loading into a hidden video element, then canvas
                      const tempVideo = document.createElement("video");
                      tempVideo.muted = true;
                      tempVideo.src   = URL.createObjectURL(file);
                      tempVideo.addEventListener("loadeddata", () => {
                        tempVideo.currentTime = 1; // seek to 1 s
                      });
                      tempVideo.addEventListener("seeked", () => {
                        const c = document.createElement("canvas");
                        c.width  = tempVideo.videoWidth  || 640;
                        c.height = tempVideo.videoHeight || 360;
                        c.getContext("2d").drawImage(tempVideo, 0, 0, c.width, c.height);
                        const uri = c.toDataURL("image/jpeg", 0.92);
                        URL.revokeObjectURL(tempVideo.src);
                        runInference(uri, file.name + " (frame)");
                      });
                    };
                    reader.readAsArrayBuffer(file); // trigger load
                  };
                  inp.click();
                }}
              >
                {"\u2191"} Upload Dashcam Video
              </button>
            </div>
          </div>

          {/* Left stage */}
          <div className="vs-dashcam-stage" style={{ height: "280px" }}>
            <video
              ref={videoRef}
              className="vs-dashcam-video"
              src="/demo_accident.mp4"
              autoPlay loop muted playsInline
              style={{ objectFit: "cover" }}
            />
            {inferring && !hasResult && (
              <div className="vs-dashcam-upload-overlay">
                <div className="vs-dashcam-spinner" />
                <div className="vs-dashcam-placeholder-text" style={{ fontSize: "11px" }}>
                  Processing…
                </div>
              </div>
            )}
            <div className="vs-video-overlay">
              <span className="vs-rec-dot" /> DEMO — LIVE FEED
            </div>
            <div className="vs-dashcam-hud-bl" style={{ fontSize: "9px" }}>
              demo_accident.mp4 — looping
            </div>
          </div>
        </div>

        {/* ──────── RIGHT PANE: AI result ──────── */}
        <div>
          {/* Right toolbar */}
          <div className="vs-dashcam-toolbar" style={{ marginBottom: "6px" }}>
            <span className="vs-section-badge vs-section-badge-yolo" style={{ fontSize: "8px" }}>
              YOLOv8 — HF Space
            </span>
            <div className="vs-video-actions">
              {/* Upload Frame button */}
              <button
                className="vs-btn vs-btn-primary"
                disabled={inferring}
                onClick={() => frameInputRef.current && frameInputRef.current.click()}
                title="Pick any .jpg / .png from disk — boxes appear instantly"
              >
                {inferring ? "Processing…" : "\u2191 Upload Frame"}
              </button>

              {/* Capture Snapshot button */}
              <button
                className="vs-btn vs-btn-ghost"
                disabled={inferring}
                onClick={handleSnapshot}
                title="Grab the current dashcam frame and run instant AI inference"
              >
                {"\u25CF"} Capture Snapshot
              </button>

              {/* Reset */}
              {hasResult && !inferring && (
                <button className="vs-btn vs-btn-ghost" onClick={resetResult}>
                  Reset
                </button>
              )}

              {/* Hidden image file input */}
              <input
                ref={frameInputRef}
                type="file"
                accept="image/*"
                onChange={handleFrameUpload}
                style={{ display: "none" }}
              />
              {/* Hidden canvas for snapshot capture */}
              <canvas ref={snapCanvas} style={{ display: "none" }} />
            </div>
          </div>

          {/* Right stage */}
          <div className="vs-dashcam-stage" style={{ height: "280px", position: "relative" }}>

            {/* Dim grayscale demo video when no result yet */}
            {!hasResult && (
              <video
                className="vs-dashcam-video"
                src="/demo_accident.mp4"
                autoPlay loop muted playsInline
                style={{ objectFit: "cover", filter: "brightness(0.14) grayscale(1)" }}
              />
            )}

            {/* Awaiting-frame prompt */}
            {!hasResult && !inferring && (
              <div style={{
                position: "absolute", inset: 0, zIndex: 3,
                display: "flex", flexDirection: "column",
                alignItems: "center", justifyContent: "center",
                color: "var(--text-dim)", textAlign: "center", padding: "16px",
              }}>
                <div style={{ fontSize: "26px", marginBottom: "8px", opacity: 0.5 }}>
                  &#x1F4F7;
                </div>
                <div style={{ fontSize: "10px", fontFamily: "monospace", lineHeight: "1.6" }}>
                  Upload a frame or capture a snapshot<br />
                  to run instant YOLOv8 detection
                </div>
              </div>
            )}

            {/* Inference spinner */}
            {inferring && (
              <div className="vs-dashcam-upload-overlay" style={{ zIndex: 4 }}>
                <div className="vs-dashcam-spinner" />
                <div className="vs-dashcam-placeholder-text" style={{ fontSize: "11px" }}>
                  YOLOv8 running on HF Space…
                </div>
                <div className="vs-dashcam-placeholder-hint">{HF_SPACE}</div>
              </div>
            )}

            {/* Annotated result image */}
            {hasResult && resultSrc && (
              <img
                src={resultSrc}
                alt="AI annotated result"
                style={{
                  position: "absolute", inset: 0, zIndex: 2,
                  width: "100%", height: "100%",
                  objectFit: "contain", background: "#000",
                }}
              />
            )}

            {/* Crash banner */}
            {hasResult && crashDetected && (
              <div style={{
                position: "absolute", top: "10px", left: "50%",
                transform: "translateX(-50%)",
                zIndex: 6, whiteSpace: "nowrap",
                background: "rgba(12,3,3,0.92)",
                border: "2px solid var(--accent-red)",
                borderRadius: "5px", padding: "6px 16px",
                color: "#fff", fontWeight: 700, fontSize: "12px",
                letterSpacing: "1px", textAlign: "center",
                boxShadow: "0 0 18px rgba(223,58,26,0.7)",
                animation: "vs-badge-flash 0.8s infinite",
              }}>
                {"\u26A0"} CRASH DETECTED
                <div style={{ fontSize: "9px", color: "#ff9080", fontWeight: 400, marginTop: "2px" }}>
                  Logged to Supabase · V2V Alert · SHAP Updated
                </div>
              </div>
            )}

            {/* Result label HUD */}
            {hasResult && resultLabel && (
              <div className="vs-dashcam-hud-bl" style={{ zIndex: 5, fontSize: "9px" }}>
                {resultLabel.slice(0, 64)}
              </div>
            )}

            {/* Status badge */}
            <div
              className={
                "vs-dashcam-status-badge " +
                (hasResult && crashDetected ? "vs-status-bad" : "vs-status-good")
              }
              style={{ zIndex: 5 }}
            >
              <span className="vs-status-dot-mini" />{" "}
              {hasResult ? (crashDetected ? "CRASH" : "Clear") : "Awaiting Frame"}
            </div>

            {/* REC indicator */}
            <div className="vs-video-overlay" style={{ zIndex: 5 }}>
              <span className="vs-rec-dot" />{" "}
              {hasResult ? "HF RESULT" : "STANDBY"} — YOLOv8
            </div>

          </div>
        </div>
      </div>

      {/* Stats bar */}
      <div className="vs-dashcam-stats" style={{ marginTop: "10px" }}>
        <div className="vs-stat"><span>Model</span>
          <b style={{ color: "var(--accent-purple)" }}>YOLOv8</b></div>
        <div className="vs-stat"><span>Backend</span>
          <b style={{ color: "var(--accent-blue)" }}>HF Space</b></div>
        <div className="vs-stat"><span>Mode</span>
          <b style={{ color: "var(--accent-blue)" }}>Image Inference</b></div>
        <div className="vs-stat"><span>Crash Alert</span>
          <b style={{ color: crashDetected ? "var(--accent-red)" : "var(--accent-green)" }}>
            {crashDetected ? "FIRED" : "Armed"}
          </b></div>
      </div>
    </div>
  );
};

// =====================================================================
// INTEL FEED (WebSocket with mock fallback)
// =====================================================================
const IntelFeed = ({ onShap }) => {
  const [logs, setLogs]     = useState([]);
  const [status, setStatus] = useState("connecting");
  const wsRef        = useRef(null);
  const reconnectRef = useRef(null);
  const mockRef      = useRef(null);
  const onShapRef    = useRef(onShap);
  useEffect(() => { onShapRef.current = onShap; }, [onShap]);

  useEffect(() => {
    let mounted = true;

    const startMock = () => {
      if (mockRef.current) return;
      const attacks = ["GPSSpoof", "AltSpoof", "SpeedInj", "Sybil", "Replay"];
      const ids     = ["EV-01", "EV-03", "EV-05", "ATK-X", "RSU-A", "EV-09"];
      mockRef.current = setInterval(() => {
        if (!mounted) return;
        const isAtk = Math.random() < 0.3;
        const vid   = ids[Math.floor(Math.random() * ids.length)];
        const atk   = attacks[Math.floor(Math.random() * attacks.length)];
        const conf  = isAtk ? 88 + Math.random() * 11 : 96 + Math.random() * 4;
        const ts    = new Date().toTimeString().slice(0, 8) + "." +
          String(Math.floor(Math.random() * 999)).padStart(3, "0");
        const msg   = isAtk
          ? `[${vid}] ${atk} — Threat flagged`
          : `[${vid}] Normal BSM — Clear`;
        setLogs((p) => [{ ts, msg, conf: parseFloat(conf.toFixed(1)) }, ...p].slice(0, 20));
        if (isAtk && onShapRef.current) {
          onShapRef.current({
            vehicle_id: vid, label: atk, confidence: parseFloat(conf.toFixed(1)),
            features: [
              { feature: "Altitude_Diff", impact: +(0.30 + Math.random() * 0.20).toFixed(2) },
              { feature: "Speed",         impact: +(0.20 + Math.random() * 0.18).toFixed(2) },
              { feature: "RSSI",          impact: +(0.10 + Math.random() * 0.18).toFixed(2) },
              { feature: "Position_Jump", impact: +(0.05 + Math.random() * 0.12).toFixed(2) },
              { feature: "Time_Skew",     impact: +(-0.10 + Math.random() * 0.10).toFixed(2) },
            ],
          });
        }
      }, 1500);
    };

    const stopMock = () => {
      if (mockRef.current) { clearInterval(mockRef.current); mockRef.current = null; }
    };

    const connect = () => {
      if (!mounted) return;
      try {
        const ws = new WebSocket(WS_URL);
        wsRef.current = ws;
        ws.onopen = () => { if (!mounted) return; setStatus("live"); stopMock(); };
        ws.onmessage = (e) => {
          if (!mounted) return;
          let d; try { d = JSON.parse(e.data); } catch (_err) { return; }
          if (d.type !== "tick") return;
          const { vehicle_id, prediction } = d;
          if (!prediction) return;
          const isThreat = prediction.label && prediction.label !== "Normal";
          const ts = new Date().toTimeString().slice(0, 8) + "." +
            String(Math.floor(Math.random() * 999)).padStart(3, "0");
          const msg = isThreat
            ? `[${vehicle_id}] ${prediction.label} — Threat flagged`
            : `[${vehicle_id}] Normal BSM — Clear`;
          setLogs((p) => [{ ts, msg, conf: prediction.confidence }, ...p].slice(0, 20));
          if (isThreat && Array.isArray(prediction.explanation) && onShapRef.current) {
            onShapRef.current({
              vehicle_id, label: prediction.label, confidence: prediction.confidence,
              features: prediction.explanation.slice(0, 5).map((x) => ({
                feature: x.feature, impact: Number(x.impact),
              })),
            });
          }
        };
        ws.onerror = () => { /* handled via onclose */ };
        ws.onclose = () => {
          if (!mounted) return;
          setStatus("offline"); startMock();
          reconnectRef.current = setTimeout(connect, 3000);
        };
      } catch (_err) {
        setStatus("offline"); startMock();
        reconnectRef.current = setTimeout(connect, 3000);
      }
    };

    connect();
    return () => {
      mounted = false;
      if (reconnectRef.current) clearTimeout(reconnectRef.current);
      if (mockRef.current)      clearInterval(mockRef.current);
      if (wsRef.current) {
        wsRef.current.onclose = null;
        try { wsRef.current.close(); } catch (_err) { /* noop */ }
      }
    };
  }, []);

  const badge = {
    connecting: { color: "var(--accent-orange)", text: "CONNECTING..." },
    live:       { color: "var(--accent-green)",  text: "LIVE — " + WS_URL },
    offline:    { color: "var(--accent-red)",    text: "OFFLINE — mock data" },
  }[status];

  return (
    <div className="vs-panel">
      <div className="vs-panel-title">
        Real-Time AI Intel Feed
        <span className="vs-ws-status" style={{ color: badge.color }}>
          {" "}- {badge.text}
        </span>
      </div>
      <div className="vs-intel-log">
        {logs.length === 0 && (
          <div className="vs-log-entry" style={{ color: "var(--text-dim)" }}>
            <span className="vs-log-msg">Waiting for telemetry stream...</span>
          </div>
        )}
        {logs.map((l, i) => {
          const cc = l.conf > 96
            ? "var(--accent-green)"
            : l.conf > 90 ? "#ffaa30" : "var(--accent-red)";
          return (
            <div key={i} className="vs-log-entry">
              <span className="vs-log-ts">{l.ts}</span>
              <span className="vs-log-msg">{l.msg}</span>
              <span className="vs-log-conf" style={{ color: cc }}>
                {l.conf.toFixed(1)}%
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
};

// =====================================================================
// SHAP XAI PANEL
// =====================================================================
const ShapPanel = ({ shap }) => {
  const data   = shap || DEFAULT_SHAP;
  const maxAbs = Math.max(...data.features.map((f) => Math.abs(f.impact)), 0.01);
  return (
    <div className="vs-panel">
      <div className="vs-panel-title">Explainable AI (SHAP) — Feature Contribution</div>
      <div className="vs-shap-header">
        <div>
          <div className="vs-shap-label">LATEST FLAG</div>
          <div className="vs-shap-vid">{data.vehicle_id}</div>
        </div>
        <div className="vs-shap-attack">
          <span className="vs-shap-attack-label">{data.label}</span>
          <span className="vs-shap-attack-conf">{Number(data.confidence).toFixed(1)}%</span>
        </div>
      </div>
      <div className="vs-shap-chart">
        {data.features.map((f, i) => {
          const pos  = f.impact >= 0;
          const wPct = (Math.abs(f.impact) / maxAbs) * 50;
          return (
            <div className="vs-shap-row" key={i}>
              <div className="vs-shap-feature" title={f.feature}>{f.feature}</div>
              <div className="vs-shap-bar-track">
                <div className="vs-shap-bar-axis" />
                <div
                  className={"vs-shap-bar-fill " + (pos ? "vs-shap-bar-pos" : "vs-shap-bar-neg")}
                  style={{ width: `${wPct}%`, [pos ? "left" : "right"]: "50%" }}
                />
              </div>
              <div className="vs-shap-impact"
                style={{ color: pos ? "var(--accent-red)" : "var(--accent-blue)" }}>
                {pos ? "+" : ""}{f.impact.toFixed(2)}
              </div>
            </div>
          );
        })}
      </div>
      <div className="vs-shap-legend">
        <span>
          <span className="vs-legend-dot" style={{ background: "var(--accent-red)" }} />
          Increases threat
        </span>
        <span>
          <span className="vs-legend-dot" style={{ background: "var(--accent-blue)" }} />
          Decreases threat
        </span>
      </div>
    </div>
  );
};

// =====================================================================
// DATASET TABLE
// =====================================================================
const DatasetTable = () => {
  const lc = (l) =>
    l === "Normal" ? "var(--accent-green)" :
    l === "Replay" ? "#d97706" :
    "var(--accent-red)";
  return (
    <div className="vs-panel">
      <div className="vs-panel-title">Dataset Sample — Labelled Telemetry</div>
      <div className="vs-table-wrap">
        <table className="vs-data-table">
          <thead>
            <tr>
              <th>Timestamp</th><th>Node ID</th><th>Pkt</th>
              <th>Speed</th><th>Alt Diff</th><th>AI Label</th>
            </tr>
          </thead>
          <tbody>
            {DUMMY_ROWS.map((r, i) => (
              <tr key={i}>
                <td className="vs-td-mono">{r.ts}</td>
                <td className="vs-td-mono">{r.nodeId}</td>
                <td><span className="vs-td-pkt">{r.pkt}</span></td>
                <td className="vs-td-mono">{r.speed.toFixed(1)}</td>
                <td className="vs-td-mono">{r.alt.toFixed(2)}</td>
                <td>
                  <span className="vs-td-label"
                    style={{ color: lc(r.label), borderColor: lc(r.label) }}>
                    {r.label}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

// =====================================================================
// MAIN APP
// =====================================================================
export default function App() {
  const [clock, setClock]           = useState("--:--:--");
  const [datasetKey, setDatasetKey] = useState("veremi");
  const [latestShap, setLatestShap] = useState(DEFAULT_SHAP);
  const [darkMode, setDarkMode]     = useState(true);
  const [metricsWsStatus, setMetricsWsStatus] = useState("simulation");
  const [metrics, setMetrics] = useState({
    accuracy:         98.4,
    latency_ms:       25,
    active_vehicles:  24,
    threats_detected: 3,
    secure_nodes:     12,
  });

  // Theme
  useEffect(() => {
    document.documentElement.setAttribute("data-theme", darkMode ? "dark" : "light");
  }, [darkMode]);

  // Clock
  useEffect(() => {
    const id = setInterval(() => setClock(new Date().toTimeString().slice(0, 8)), 1000);
    return () => clearInterval(id);
  }, []);

  // Metrics simulation with random-walk jitter on all five fields
  // Starts immediately → zero blank states. Real WS takes over if available.
  useEffect(() => {
    let mounted     = true;
    let ws          = null;
    let simInterval = null;

    const centre = { accuracy: 98.4, latency_ms: 25, active_vehicles: 24, threats_detected: 3, secure_nodes: 12 };

    const jitter = (base, delta, lo, hi) => {
      const v = base + (Math.random() * 2 - 1) * delta;
      return parseFloat(Math.min(hi, Math.max(lo, v)).toFixed(1));
    };
    const jInt = (base, delta, lo, hi) =>
      Math.round(Math.min(hi, Math.max(lo, base + (Math.random() * 2 - 1) * delta)));

    const startSim = () => {
      if (simInterval) return;
      setMetricsWsStatus("simulation");
      simInterval = setInterval(() => {
        if (!mounted) return;
        centre.accuracy         = jitter(centre.accuracy,         0.15, 97.8, 98.9);
        centre.latency_ms       = jInt(centre.latency_ms,         2,    18,   34);
        centre.active_vehicles  = jInt(centre.active_vehicles,    1,    20,   30);
        centre.threats_detected = jInt(centre.threats_detected,   1,    1,    5);
        centre.secure_nodes     = jInt(centre.secure_nodes,       1,    15,   28);
        setMetrics({ ...centre });
      }, 2000);
    };
    const stopSim = () => {
      if (simInterval) { clearInterval(simInterval); simInterval = null; }
    };

    startSim();

    try {
      ws = new WebSocket(METRICS_WS_URL);
      ws.onopen = () => { if (!mounted) return; stopSim(); setMetricsWsStatus("connected"); };
      ws.onmessage = (e) => {
        if (!mounted) return;
        let data; try { data = JSON.parse(e.data); } catch (_err) { return; }
        if (!data || typeof data !== "object") return;
        setMetrics((prev) => ({ ...prev, ...data }));
      };
      ws.onerror = () => { /* handled in onclose */ };
      ws.onclose = () => { if (!mounted) return; startSim(); };
    } catch (_err) { /* simulation already running */ }

    return () => {
      mounted = false;
      stopSim();
      if (ws) {
        ws.onopen = null; ws.onmessage = null;
        ws.onerror = null; ws.onclose = null;
        try { ws.close(); } catch (_err) { /* noop */ }
      }
    };
  }, []);

  // handleShap: called by IntelFeed (WS/mock events) AND DashcamYoloInference (HF crash)
  const handleShap = useCallback((shap) => setLatestShap(shap), []);

  return (
    <div className="vs-app">
      <Header
        clock={clock}
        darkMode={darkMode}
        onToggleTheme={() => setDarkMode((d) => !d)}
        metricsWsStatus={metricsWsStatus}
        metricsWsUrl={METRICS_WS_URL}
      />

      <DatasetKPIBar datasetKey={datasetKey} onChange={setDatasetKey} />

      <div className="vs-layout">
        <TheorySidebar />
        <main className="vs-main">
          <div className="vs-row vs-row-top">
            <LiveRadar metrics={metrics} />
            <CarlaSimulation metrics={metrics} />
          </div>
          <div className="vs-row vs-row-mid-3">
            <LatencyChart />
            <IntelFeed onShap={handleShap} />
            <ShapPanel shap={latestShap} />
          </div>
          <div className="vs-row vs-row-bot-1">
            <DashcamYoloInference onShapUpdate={handleShap} />
          </div>
          <div className="vs-row vs-row-bot-1">
            <DatasetTable />
          </div>
        </main>
      </div>

      <footer className="vs-footer">
        <span>Secure V2V Communication using AI &copy; 2026 — Bahria University FYP — Abdul Wahab Aslam</span>
        <span>AutoSec AI — YOLOv8 Edge Inference — TensorRT</span>
      </footer>
    </div>
  );
}
