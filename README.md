# Smart City Intelligence & Dynamic Traffic System (SIH 2025)

An end-to-end intelligent transportation and urban infrastructure system integrating **YOLOv8 Edge Computer Vision**, **Webster's Optimum Cycle Dynamic Traffic Signal Optimization**, **Emergency Vehicle Priority Override ("Green Corridor")**, **Road Damage Surveillance & Heuristic Prioritization**, a **FastAPI backend with security hardening**, and an interactive **React dashboard**.

> **Project status**: working prototype (hackathon build). The ML pipeline, decision engine, API and dashboard are implemented and covered by 50 automated tests. See [Known limitations](#known-limitations) for what is **not** validated yet.

---

## Key Features (Smart India Hackathon 2025 Architecture)

1. **AI Dynamic Traffic Light Optimization**:
   - Converts detected vehicle counts into **PCU (Passenger Car Units)** per Indian Roads Congress standards (IRC: 106-1990).
   - Dynamically calculates optimal intersection cycle length and green splits via **Webster’s Optimum Cycle Formula**:
     $$C_o = \frac{1.5 L + 5}{1 - Y}$$
   - Reports an *estimated* delay/fuel/CO2 saving versus an equal-split fixed-time plan, computed analytically with Webster's uniform-delay formula (`basis: "analytical_webster_uniform_delay"` in the API response). It runs on flows estimated from detected counts, so it is **not** a simulation or field measurement. When demand exceeds capacity (`benefits.oversaturated: true`) the formula is invalid and the savings fields are `null` instead of a made-up number.

2. **Emergency Vehicle Priority ("Green Corridor")**:
   - Emergency presence is an **input** to the signal engine (`has_emergency` flag per approach, or an `ambulance` / `fire_truck` / `police` class in the detections). The trained vehicle model only knows car, van, truck and tram, so it does **not** detect ambulances. A colour-based beacon heuristic exists only in the optional demo mode (see below) and must not be relied on.
   - When flagged, the engine preempts normal cycle to activate an uninterrupted **Green Corridor** while putting conflicting lanes on safe all-red hold. If several approaches are flagged at once they are served one after another (`emergency_approaches`, 40 s each).
   - Pushes instantaneous critical alerts to city traffic command controllers.

3. **Road Damage AI Surveillance & Repair Prioritization**:
   - Detects potholes, alligator cracks, longitudinal and transverse cracks (trained on RDD2022 India).
   - Computes composite municipal priority by combining damage severity ($0-100$) and segment traffic importance ($0.1-5.0$):
     $$\text{Priority} = \text{Damage Score} \times \text{clamp}\!\left(\tfrac{\text{Traffic Importance}}{2},\,0.2,\,2.5\right) \times (1 + 0.18\,P + 0.08\,A + 0.03\,C)$$
     where $P$ = potholes, $A$ = alligator cracks, $C$ = longitudinal + transverse cracks; the result is capped at 100.

4. **Security hardening** (see `backend/app/core/security.py`):
   - **Headers**: `Content-Security-Policy` (locked down for the API, relaxed only for `/docs` and `/redoc`), `Strict-Transport-Security` (effective over HTTPS), `X-Frame-Options: DENY`, `X-Content-Type-Options: nosniff`, `Referrer-Policy`, `Permissions-Policy`.
   - **Rate limiting**: sliding-window per-IP limit (240 req/min by default). It is in-memory, so it is per-process; use a reverse proxy or Redis for multi-worker deployments.
   - **Upload validation**: size limit plus file-signature (magic byte) check for JPEG, PNG and WEBP before decoding.
   - **Optional API key**: set `REQUIRE_API_KEY=true` and `API_KEY=<long random value>` to require an `X-API-Key` header on every POST/PATCH endpoint. No default key is shipped.
   - **SQL injection**: all queries go through the SQLAlchemy ORM (parameterised).
   - This is sensible hardening for a prototype, not a formal OWASP audit.

5. **Interactive React Control Dashboard**:
   - Real-time 4-way intersection visualizer with animated LED traffic light indicators and live countdown timers.
   - One-click SIH real-world demonstration scenarios (Morning Rush, Emergency Ambulance Inbound, Commercial Freight Gridlock, Balanced Flow).
   - Municipal road repair priority dispatch board and model performance analytics.

---

## Quickstart

### 1. Python Environment & Dependencies
```bash
# Activate virtualenv or system python (Python 3.10 - 3.13)
pip install -r requirements.txt
```

### 2. Model weights
Weights are not stored in git. Put `vehicle_best.pt` and `road_damage_best.pt` in `models/` (download or train them: see [`models/README.md`](models/README.md)). Without them the prediction endpoints return **HTTP 503**; for demos only you can set `ALLOW_HEURISTIC_FALLBACK=true` (copy `.env.example` to `.env`) to use a clearly-labelled classical-CV heuristic.

### 3. Run Backend (FastAPI)
```bash
uvicorn backend.app.main:app --reload --port 8000
# Interactive API documentation: http://localhost:8000/docs
```

### 4. Run Frontend Dashboard (React + Vite)
```bash
cd frontend
npm install
npm run dev
# Open in browser: http://localhost:5173
```

---

## API Endpoints Summary

| Endpoint | Method | Purpose |
|---|---|---|
| `/signals/optimize-dynamic` | `POST` | Calculates Webster optimum cycle, green splits, and delay savings for input approaches |
| `/intersections/{id}/optimize-signal` | `POST` | Computes signal timings, saves `SignalRecommendation`, and triggers Green Corridor alerts |
| `/signals/latest` | `GET` | Retrieves recent signal timing schedules |
| `/segments/prioritize-repairs` | `POST` | Ranks all city road segments by urgency of repair |
| `/predict/vehicles` | `POST` | Upload image -> YOLO vehicle detection + congestion estimation + annotated image |
| `/predict/road-damage` | `POST` | Upload image -> Pothole/crack detection + severity score |
| `/alerts` | `GET` | Live feed of emergency corridor triggers, congestion spikes, and hazardous potholes |
| `/alerts/{id}/acknowledge` | `POST` | Acknowledge/dismiss active alerts |
| `/maintenance` | `GET/POST/PATCH`| Municipal road maintenance work order lifecycle management |
| `/model-metrics` | `GET` | Real evaluation benchmark metrics (mAP50, precision, recall, latency) |
| `/health` | `GET` | Database/model status (`ok` or `degraded` when weights are missing) and security settings |

---

## Running Unit Tests
```bash
pytest tests -v
```
The suite (50 tests) covers the signal engine, repair ranking, API behaviour, upload validation, security headers, API-key enforcement, rate limiting, ONNX decoding and the missing-weights behaviour. CI runs it on every push (`.github/workflows/ci.yml`).

---

## Known limitations
- **Road damage detector is weak**: mAP50 is about 0.36 on the held-out test split (recall about 0.33; transverse cracks 0.16 on only 16 test instances). Treat its output as a screening aid, not an inspection result.
- **Congestion target is a proxy** computed from annotated vehicle boxes, not human-rated congestion (see `reports/RESULTS.md`).
- **Signal savings are estimates**, not simulated. Flow is derived from detected counts with a fixed scaling factor. Validate with a traffic simulator (e.g. SUMO) or field data before quoting numbers.
- **No emergency-vehicle model**: see Emergency Vehicle Priority above.
- **Demo mode** (`ALLOW_HEURISTIC_FALLBACK=true`) is not a trained model and its accuracy has not been measured.

---

## Project Structure
```
backend/app/
  api/routes.py            # REST endpoints including Webster signals & repair rankings
  api/schemas.py           # Pydantic validation models
  core/security.py         # Security headers, rate limiting, upload validation, API-key check
  services/model_service.py # Model loading (ONNX/PyTorch), 503 when weights are missing
  db/models.py             # SQLAlchemy 2.0 relational database tables
decision_engine/
  traffic_signal.py        # Webster's optimum cycle, PCU calculation & Emergency Green Corridor
  repair_priority.py       # Heuristic road repair composite ranking algorithm
  scoring.py               # Pavement damage heuristic scoring
frontend/src/
  App.jsx                  # Interactive 4-Way junction simulator, emergency triggers, & dashboards
  index.css                # Glassmorphic dark UI & LED traffic light animations
ml/
  preprocessing/           # KITTI & RDD2022 dataset preparation & augmentation
  training/                # YOLOv8 & Congestion Random Forest training pipelines
  evaluation/              # mAP, precision, recall, and latency benchmarks
reports/
  RESULTS.md               # Evaluated model metrics, curves, and confusion matrices
models/README.md           # Where to get/put the model weights
tests/                     # Automated pytest test suite
```
