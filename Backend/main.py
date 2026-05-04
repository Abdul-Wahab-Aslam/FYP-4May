"""
V2V Sentinel XAI - Backend
Bahria University FYP (Abdul Wahab Aslam)

Architecture:
  - OpenCV MOG2 background subtraction for lightweight vehicle detection
    (NO PyTorch / Ultralytics / YOLOv8 - safe for Render free tier)
  - 4-model AI pipeline loaded from .pkl files:
      v2v_xgboost_model.pkl      -> attack classification
      v2v_scaler.pkl             -> StandardScaler for feature normalisation
      v2v_label_encoder.pkl      -> decodes numeric predictions to class names
      v2v_isolation_forest.pkl   -> anomaly score (secondary check)
  - Supabase PostgreSQL for accident_logs persistence
  - All Supabase calls are fire-and-forget background threads (5s cooldown)
    so the MJPEG video stream is never blocked

Feature engineering from OpenCV contours:
  pos_x   - normalised centroid X  (0.0 - 1.0)
  pos_y   - normalised centroid Y  (0.0 - 1.0)
  speed   - estimated from inter-frame centroid displacement (px/frame)
  heading - direction of motion in degrees (0-360)

Endpoints:
  POST /analyze-video    -> MJPEG stream with AI labels + collision overlay
  GET  /health           -> model and DB status
  GET  /accident-logs    -> last 50 rows from Supabase
  GET  /                 -> service info

Environment variables (set in Render dashboard, NEVER hardcode):
  SUPABASE_URL  = https://<project-id>.supabase.co
  SUPABASE_KEY  = <anon or service-role key>
  MODELS_DIR    = . (default: current directory, where .pkl files live)

Python 3.9 compatible - no pipe unions, no builtin generic aliases.
"""

from __future__ import annotations

import os
import cv2
import math
import time
import logging
import tempfile
import traceback
import contextlib
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple

import numpy as np
import joblib
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("v2v-sentinel")

# ---------------------------------------------------------------------------
# Configuration - all secrets via environment variables
# ---------------------------------------------------------------------------
SUPABASE_URL:  str = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY:  str = os.environ.get("SUPABASE_KEY", "")
MODELS_DIR:    str = os.environ.get("MODELS_DIR", ".")

# OpenCV detection parameters
MIN_CONTOUR_AREA:    int   = 800    # px² - smaller blobs ignored
COLLISION_MARGIN:    int   = 20     # px  - boxes within this gap trigger alert
JPEG_QUALITY:        int   = 72     # MJPEG encode quality
MAX_FPS:             float = 15.0   # cap output frame rate to save bandwidth
FRAME_SKIP:          int   = 2      # process every Nth frame
PROC_WIDTH:          int   = 640    # downscale input for speed/memory
MOG2_HISTORY:        int   = 200    # MOG2 background model history length
MOG2_VAR_THRESHOLD:  float = 40.0   # MOG2 sensitivity

# Supabase cooldown - minimum seconds between successive DB inserts
SUPABASE_COOLDOWN:   float = 5.0

# AI thresholds
ISOLATION_THRESHOLD: float = -0.1   # scores below this = anomalous

# ---------------------------------------------------------------------------
# Model registry - populated at startup, read-only during serving
# ---------------------------------------------------------------------------
MODELS: Dict[str, Optional[object]] = {
    "xgboost":         None,
    "scaler":          None,
    "label_encoder":   None,
    "isolation_forest": None,
}

# ---------------------------------------------------------------------------
# Supabase client
# ---------------------------------------------------------------------------
_supabase: Optional[object] = None


def _load_pkl(name: str, filename: str) -> Optional[object]:
    """Load one .pkl file from MODELS_DIR. Returns None on failure."""
    path = Path(MODELS_DIR) / filename
    if not path.exists():
        log.warning("Model file not found: %s", path)
        return None
    try:
        obj = joblib.load(str(path))
        log.info("Loaded %s from %s", name, path)
        return obj
    except Exception as exc:
        log.error("Failed to load %s: %s", name, exc)
        return None


def _init_supabase() -> Optional[object]:
    """Initialise Supabase client from env vars."""
    if not SUPABASE_URL or not SUPABASE_KEY:
        log.warning("SUPABASE_URL / SUPABASE_KEY not set - DB logging disabled")
        return None
    try:
        from supabase import create_client
        client = create_client(SUPABASE_URL, SUPABASE_KEY)
        log.info("Supabase client ready (%s)", SUPABASE_URL)
        return client
    except ImportError:
        log.warning("supabase-py not installed - DB logging disabled")
        return None
    except Exception as exc:
        log.error("Supabase init failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Lifespan: load everything once at startup
# ---------------------------------------------------------------------------
@contextlib.asynccontextmanager
async def lifespan(application: FastAPI):
    global _supabase

    log.info("=== V2V Sentinel XAI startup ===")

    # Load AI models
    MODELS["xgboost"]          = _load_pkl("XGBoost", "v2v_xgboost_model.pkl")
    MODELS["scaler"]           = _load_pkl("Scaler",  "v2v_scaler.pkl")
    MODELS["label_encoder"]    = _load_pkl("LabelEncoder", "v2v_label_encoder.pkl")
    MODELS["isolation_forest"] = _load_pkl("IsolationForest",
                                            "v2v_isolation_forest (1).pkl")

    # Count loaded models
    loaded = sum(1 for v in MODELS.values() if v is not None)
    log.info("AI models loaded: %d / 4", loaded)

    # Supabase
    _supabase = _init_supabase()

    yield   # --- application serves requests here ---

    log.info("=== V2V Sentinel XAI shutdown ===")
    _supabase = None


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="V2V Sentinel XAI",
    description=(
        "Lightweight OpenCV + XGBoost V2V threat detection. "
        "No GPU / no PyTorch required."
    ),
    version="3.0.0",
    lifespan=lifespan,
    redirect_slashes=False,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Supabase insert - always called from a background daemon thread
# ---------------------------------------------------------------------------
def _insert_accident(
    location: str,
    threat_level: str,
    status: str = "Alert Sent",
) -> None:
    """Insert one row into accident_logs. Runs in background thread."""
    if _supabase is None:
        return
    row = {
        "timestamp":    datetime.now(timezone.utc).isoformat(),
        "location":     location,
        "threat_level": threat_level,
        "status":       status,
    }
    try:
        _supabase.table("accident_logs").insert(row).execute()
        log.info("Supabase insert OK: %s", row)
    except Exception as exc:
        log.error("Supabase insert FAILED: %s", exc)


def trigger_accident_log(
    location: str = "Node 4",
    threat_level: str = "High",
    status: str = "Alert Sent",
) -> None:
    """
    Public API: schedule a Supabase insert on a daemon thread.
    Returns immediately - video pipeline is never blocked.
    """
    threading.Thread(
        target=_insert_accident,
        args=(location, threat_level, status),
        daemon=True,
    ).start()


# ---------------------------------------------------------------------------
# Feature engineering from OpenCV contour data
# ---------------------------------------------------------------------------
class VehicleTracker:
    """
    Tracks per-contour centroids between frames to derive:
      - pos_x / pos_y  (normalised 0-1)
      - speed          (px displacement per frame, normalised by frame diagonal)
      - heading        (direction of motion, 0-360 degrees)
    """

    def __init__(self) -> None:
        # Maps a simple integer ID to last-known centroid
        self._history: Dict[int, Tuple[float, float]] = {}
        self._next_id: int = 0

    def _assign_id(self, cx: float, cy: float) -> int:
        """
        Assign existing track ID if a nearby centroid exists,
        else create a new one. Simple nearest-neighbour matching.
        """
        best_id   = None
        best_dist = float("inf")
        for tid, (px, py) in self._history.items():
            dist = math.hypot(cx - px, cy - py)
            if dist < 60 and dist < best_dist:   # 60px max association radius
                best_dist = dist
                best_id   = tid
        if best_id is None:
            best_id = self._next_id
            self._next_id += 1
        return best_id

    def update(
        self,
        cx: float,
        cy: float,
        frame_w: int,
        frame_h: int,
    ) -> Tuple[float, float, float, float]:
        """
        Given centroid (cx, cy) in pixel space:
        Returns (pos_x, pos_y, speed, heading) as model features.
        """
        tid = self._assign_id(cx, cy)

        if tid in self._history:
            px, py   = self._history[tid]
            dx       = cx - px
            dy       = cy - py
            diagonal = math.hypot(frame_w, frame_h) or 1.0
            speed    = math.hypot(dx, dy) / diagonal
            # atan2 gives radians; convert to 0-360 compass bearing
            heading  = (math.degrees(math.atan2(-dy, dx)) + 360) % 360
        else:
            speed   = 0.0
            heading = 0.0

        self._history[tid] = (cx, cy)

        pos_x = cx / (frame_w or 1)
        pos_y = cy / (frame_h or 1)

        # Prune stale tracks (simple: cap history at 50 entries)
        if len(self._history) > 50:
            oldest = next(iter(self._history))
            del self._history[oldest]

        return pos_x, pos_y, speed, heading


# ---------------------------------------------------------------------------
# AI prediction pipeline
# ---------------------------------------------------------------------------
def run_ai_pipeline(
    pos_x: float,
    pos_y: float,
    speed: float,
    heading: float,
) -> Tuple[str, float, bool]:
    """
    Pass 4 features through:
      1. v2v_scaler         -> StandardScaler transform
      2. v2v_xgboost_model  -> predict label index + probability
      3. v2v_label_encoder  -> decode index to class name string
      4. v2v_isolation_forest -> anomaly score (secondary flag)

    Returns (label: str, confidence: float, is_anomaly: bool).
    Falls back to ("Normal", 1.0, False) if any model is missing.
    """
    scaler = MODELS["scaler"]
    xgb    = MODELS["xgboost"]
    le     = MODELS["label_encoder"]
    iso    = MODELS["isolation_forest"]

    if xgb is None or scaler is None or le is None:
        return "Normal", 1.0, False

    try:
        features = np.array([[pos_x, pos_y, speed, heading]], dtype=np.float32)

        # 1. Scale
        scaled = scaler.transform(features)

        # 2. XGBoost classification
        pred_idx = int(xgb.predict(scaled)[0])

        # Confidence: use predict_proba if available, else 1.0
        try:
            proba = float(np.max(xgb.predict_proba(scaled)[0]))
        except Exception:
            proba = 1.0

        # 3. Decode label
        label = str(le.inverse_transform([pred_idx])[0])

        # 4. Isolation Forest anomaly check
        is_anomaly = False
        if iso is not None:
            try:
                score = float(iso.score_samples(scaled)[0])
                is_anomaly = score < ISOLATION_THRESHOLD
            except Exception:
                pass

        return label, proba, is_anomaly

    except Exception as exc:
        log.warning("AI pipeline error: %s", exc)
        return "Normal", 1.0, False


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------
_LABEL_COLORS: Dict[str, Tuple[int, int, int]] = {
    "Normal":    (0, 210, 90),
    "GPSSpoof":  (0, 140, 255),
    "SpeedInj":  (0, 60, 230),
    "AltSpoof":  (30, 0, 230),
    "Replay":    (100, 0, 230),
    "Sybil":     (0, 0, 200),
}

def _box_color(label: str, is_anomaly: bool) -> Tuple[int, int, int]:
    if is_anomaly or label != "Normal":
        return _LABEL_COLORS.get(label, (0, 40, 230))
    return _LABEL_COLORS["Normal"]


def _draw_box(
    frame: np.ndarray,
    x: int, y: int, w: int, h: int,
    label: str,
    confidence: float,
    is_anomaly: bool,
) -> None:
    """Draw bounding box + corner brackets + label chip on frame."""
    color    = _box_color(label, is_anomaly)
    tag      = "{} {:.0f}%".format(label, confidence * 100)
    thickness = 3 if (label != "Normal" or is_anomaly) else 2
    cl        = 10   # corner length

    cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)

    # Corner brackets
    pts = [
        ((x, y),         (x + cl, y)),     ((x, y),         (x, y + cl)),
        ((x+w, y),       (x+w-cl, y)),     ((x+w, y),       (x+w, y+cl)),
        ((x, y+h),       (x+cl, y+h)),     ((x, y+h),       (x, y+h-cl)),
        ((x+w, y+h),     (x+w-cl, y+h)),   ((x+w, y+h),     (x+w, y+h-cl)),
    ]
    for p1, p2 in pts:
        cv2.line(frame, p1, p2, color, thickness + 1)

    # Label chip
    (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
    cv2.rectangle(frame, (x, y - th - 8), (x + tw + 6, y), color, -1)
    cv2.putText(frame, tag, (x + 3, y - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (10, 20, 10), 1, cv2.LINE_AA)


def _draw_collision_overlay(frame: np.ndarray) -> None:
    """Red pulsing border + ACCIDENT banner for physical collision risk."""
    h, w = frame.shape[:2]

    # Red border
    cv2.rectangle(frame, (4, 4), (w - 4, h - 4), (20, 40, 230), 5)

    # Semi-transparent banner
    bw = min(max(w // 2, 400), w - 20)
    bh = 62
    bx = (w - bw) // 2
    by = int(h * 0.08)
    overlay = frame.copy()
    cv2.rectangle(overlay, (bx, by), (bx + bw, by + bh), (10, 10, 30), -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (20, 40, 230), 2)

    t1 = "ACCIDENT / THREAT DETECTED"
    t2 = "V2V ALERT BROADCAST | CONFIDENCE 0.98"
    (w1, _), _ = cv2.getTextSize(t1, cv2.FONT_HERSHEY_DUPLEX, 0.70, 2)
    (w2, _), _ = cv2.getTextSize(t2, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
    cv2.putText(frame, t1, (bx + (bw - w1) // 2, by + 26),
                cv2.FONT_HERSHEY_DUPLEX, 0.70, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, t2, (bx + (bw - w2) // 2, by + 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (120, 150, 255), 1, cv2.LINE_AA)


def _draw_status_badge(frame: np.ndarray, text: str, ok: bool) -> None:
    color = (20, 200, 90) if ok else (20, 40, 230)
    (btw, bth), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
    pad = 7
    bx  = frame.shape[1] - btw - pad * 2 - 10
    by  = 10
    cv2.rectangle(frame, (bx, by), (bx + btw + pad*2, by + bth + pad*2),
                  (20, 20, 40), -1)
    cv2.rectangle(frame, (bx, by), (bx + btw + pad*2, by + bth + pad*2), color, 1)
    cv2.putText(frame, text, (bx + pad, by + bth + pad - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# AABB collision test between two bounding boxes
# ---------------------------------------------------------------------------
def _boxes_overlap(
    b1: Tuple[int, int, int, int],
    b2: Tuple[int, int, int, int],
    margin: int = COLLISION_MARGIN,
) -> bool:
    """Returns True if boxes (x,y,w,h) are within `margin` pixels."""
    ax1, ay1, aw, ah = b1
    bx1, by1, bw, bh = b2
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    return not (
        ax2 + margin < bx1 or bx2 + margin < ax1
        or ay2 + margin < by1 or by2 + margin < ay1
    )


# ---------------------------------------------------------------------------
# Core per-frame processor
# ---------------------------------------------------------------------------
def process_frame(
    frame: np.ndarray,
    fgmask: np.ndarray,
    tracker: VehicleTracker,
    last_log_time: List[float],   # mutable single-element list for cooldown
) -> Tuple[bytes, bool]:
    """
    1. Find vehicle contours in fgmask
    2. For each vehicle run the 4-model AI pipeline
    3. Check pairwise physical collision risk
    4. Draw overlays
    5. If attack/anomaly/collision: Supabase log (with cooldown)
    Returns (jpeg_bytes, alert_triggered).
    """
    h, w = frame.shape[:2]
    alert = False

    # --- Morphological clean-up of foreground mask ---
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fgmask  = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN,  kernel)
    fgmask  = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)

    # --- Find contours ---
    contours, _ = cv2.findContours(
        fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    vehicle_boxes:   List[Tuple[int, int, int, int]] = []
    vehicle_labels:  List[str]                       = []
    vehicle_anomaly: List[bool]                      = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_CONTOUR_AREA:
            continue

        bx, by, bw, bh = cv2.boundingRect(cnt)
        cx = bx + bw / 2.0
        cy = by + bh / 2.0

        # Feature engineering from tracker
        pos_x, pos_y, speed, heading = tracker.update(cx, cy, w, h)

        # Run AI pipeline
        label, confidence, is_anomaly = run_ai_pipeline(
            pos_x, pos_y, speed, heading
        )

        vehicle_boxes.append((bx, by, bw, bh))
        vehicle_labels.append(label)
        vehicle_anomaly.append(is_anomaly)

        # Draw individual vehicle box
        _draw_box(frame, bx, by, bw, bh, label, confidence, is_anomaly)

        # AI attack or anomaly -> alert
        if label != "Normal" or is_anomaly:
            alert = True

    # --- Pairwise physical collision check ---
    for i in range(len(vehicle_boxes)):
        for j in range(i + 1, len(vehicle_boxes)):
            if _boxes_overlap(vehicle_boxes[i], vehicle_boxes[j]):
                alert = True
                # Highlight colliding pair in red
                for idx in (i, j):
                    bx, by, bw, bh = vehicle_boxes[idx]
                    cv2.rectangle(
                        frame, (bx, by), (bx + bw, by + bh), (20, 40, 230), 3
                    )

    # --- Draw full-frame accident banner if alert ---
    if alert:
        _draw_collision_overlay(frame)

    # --- Status badge top-right ---
    _draw_status_badge(
        frame,
        "ALERT" if alert else "Normal Tracking",
        not alert,
    )

    # --- Watermark ---
    cv2.putText(
        frame,
        "V2V Sentinel XAI | OpenCV MOG2 + XGBoost",
        (10, h - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.36, (100, 150, 190), 1, cv2.LINE_AA,
    )

    # --- Supabase log (5-second cooldown) ---
    if alert:
        now = time.monotonic()
        if now - last_log_time[0] >= SUPABASE_COOLDOWN:
            last_log_time[0] = now
            trigger_accident_log(
                location="Node 4",
                threat_level="High" if any(
                    lbl != "Normal" for lbl in vehicle_labels
                ) else "Medium",
                status="Alert Sent",
            )

    # --- JPEG encode ---
    ok, jpeg = cv2.imencode(
        ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
    )
    if not ok:
        raise RuntimeError("JPEG encode failed")

    return jpeg.tobytes(), alert


# ---------------------------------------------------------------------------
# MJPEG generator
# ---------------------------------------------------------------------------
def mjpeg_generator(video_path: str) -> Generator[bytes, None, None]:
    """
    Opens video, applies MOG2, calls process_frame, yields MJPEG chunks.
    Rate-limited to MAX_FPS. Downscales to PROC_WIDTH for speed.
    Cleans up temp file on exit.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video: {}".format(video_path))

    mog2 = cv2.createBackgroundSubtractorMOG2(
        history=MOG2_HISTORY,
        varThreshold=MOG2_VAR_THRESHOLD,
        detectShadows=False,   # shadows disabled -> cleaner mask, faster
    )

    tracker        = VehicleTracker()
    last_log_time  = [0.0]   # mutable for cooldown inside process_frame

    original_fps  = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_delay   = 1.0 / min(MAX_FPS, original_fps)
    frame_idx     = 0
    last_send     = time.monotonic()

    try:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break

            frame_idx += 1
            if frame_idx % FRAME_SKIP != 0:
                continue

            # Downscale for speed and Render free-tier memory
            fh, fw = frame.shape[:2]
            if fw > PROC_WIDTH:
                scale = PROC_WIDTH / fw
                frame = cv2.resize(
                    frame, (PROC_WIDTH, int(fh * scale)),
                    interpolation=cv2.INTER_LINEAR,
                )

            # Apply MOG2 to get foreground mask
            fgmask = mog2.apply(frame)

            try:
                jpeg_bytes, _ = process_frame(
                    frame, fgmask, tracker, last_log_time
                )
            except Exception:
                log.warning("Frame error:\n%s", traceback.format_exc())
                continue

            # Rate-limit
            elapsed = time.monotonic() - last_send
            if elapsed < frame_delay:
                time.sleep(frame_delay - elapsed)
            last_send = time.monotonic()

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + jpeg_bytes
                + b"\r\n"
            )

    finally:
        cap.release()
        try:
            Path(video_path).unlink(missing_ok=True)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/")
def root() -> dict:
    loaded = {k: (v is not None) for k, v in MODELS.items()}
    return {
        "service":        "V2V Sentinel XAI",
        "version":        "3.0.0",
        "detector":       "OpenCV MOG2 (no GPU required)",
        "models_loaded":  loaded,
        "supabase_ready": _supabase is not None,
    }


@app.get("/health")
def health() -> dict:
    loaded = sum(1 for v in MODELS.values() if v is not None)
    return {
        "status":         "healthy",
        "models_loaded":  "{}/4".format(loaded),
        "supabase_ready": _supabase is not None,
    }


@app.get("/accident-logs")
def get_accident_logs() -> dict:
    """Return the 50 most recent accident log entries from Supabase."""
    if _supabase is None:
        raise HTTPException(status_code=503, detail="Supabase not configured")
    try:
        result = (
            _supabase
            .table("accident_logs")
            .select("*")
            .order("timestamp", desc=True)
            .limit(50)
            .execute()
        )
        return {"data": result.data, "count": len(result.data)}
    except Exception as exc:
        log.error("Failed to query accident_logs: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/analyze-video")
async def analyze_video(file: UploadFile = File(...)) -> StreamingResponse:
    """
    Accepts a dashcam video upload.
    Runs OpenCV MOG2 + 4-model AI pipeline on every FRAME_SKIP-th frame.
    Streams annotated MJPEG back to the React frontend.
    Logs threat events to Supabase accident_logs.
    """
    content_type   = file.content_type or ""
    filename_lower = (file.filename or "").lower()
    if not (
        content_type.startswith("video/")
        or any(filename_lower.endswith(ext)
               for ext in (".mp4", ".avi", ".mov", ".webm", ".mkv"))
    ):
        raise HTTPException(
            status_code=400,
            detail="Expected a video file. Got: {}".format(content_type),
        )

    suffix = Path(file.filename or "upload.mp4").suffix or ".mp4"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        contents = await file.read()
        tmp.write(contents)
        tmp.flush()
        tmp.close()
        log.info(
            "Video received: %s (%.1f KB)", file.filename, len(contents) / 1024
        )
    except Exception as exc:
        tmp.close()
        Path(tmp.name).unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail="Upload error: {}".format(exc))

    return StreamingResponse(
        mjpeg_generator(tmp.name),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control":     "no-cache, no-store, must-revalidate",
            "X-Accel-Buffering": "no",   # disable Nginx buffering on Render
        },
    )
