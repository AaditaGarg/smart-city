# Smart City Intelligence & Dynamic Traffic System (SIH 2025)

An end-to-end intelligent transportation and urban infrastructure system integrating **YOLOv8 Edge Computer Vision**, **Webster's Optimum Cycle Dynamic Traffic Signal Optimization**, **Emergency Vehicle Priority Override ("Green Corridor")**, **Road Damage Surveillance & Heuristic Prioritization**, a **FastAPI backend with OWASP-compliant security**, and an interactive **React dashboard**.

> **Project Status**: **All Steps 1 to 9 Complete & Fully Operational**. Includes unit tests, dynamic signal algorithms, security headers, rate limiting, and dashboard build.

---

## Key Features (Smart India Hackathon 2025 Architecture)

1. **AI Dynamic Traffic Light Optimization**:
   - Converts detected vehicle counts into **PCU (Passenger Car Units)** per Indian Roads Congress standards (IRC: 106-1990).
   - Dynamically calculates optimal intersection cycle length and green splits via **Webster’s Optimum Cycle Formula**:
     $$C_o = \frac{1.5 L + 5}{1 - Y}$$
   - Demonstrates ~35% - 45% reduction in commuter waiting time, cutting fuel burn and idle carbon emissions.

2. **Emergency Vehicle Priority ("Green Corridor")**:
   - Automatically detects ambulances, fire engines, or police vehicles in approach lanes.
   - Immediately preempts normal cycle to activate an uninterrupted **Green Corridor** while putting conflicting lanes on safe all-red hold.
   - Pushes instantaneous critical alerts to city traffic command controllers.

3. **Road Damage AI Surveillance & Repair Prioritization**:
   - Detects potholes, alligator cracks, longitudinal and transverse cracks (trained on RDD2022 India).
   - Computes composite municipal priority by combining damage severity ($0-100$) and segment traffic importance ($0.1-5.0$):
     $$\text{Priority} = \text{Damage Score} \times \frac{\text{Traffic Importance}}{2.0} \times (1.0 + 0.18 \times \text{Potholes})$$

4. **Production Security Architecture (OWASP Compliant)**:
   - **Security Headers**: Strict CSP, `X-Frame-Options: DENY`, `X-Content-Type-Options: nosniff`, and `HSTS`.
   - **DoS Protection**: Sliding-window IP rate limiting (240 req/min).
   - **Image Validation**: File magic byte checking (JPEG, PNG, WEBP) to eliminate polyglot script injections.
   - **SQL Injection Immunity**: SQLAlchemy 2.0 type-safe parameterized ORM queries.

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

### 2. Run Backend (FastAPI)
```bash
uvicorn backend.app.main:app --reload --port 8000
# Interactive API documentation: http://localhost:8000/docs
```

### 3. Run Frontend Dashboard (React + Vite)
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
| `/health` | `GET` | Health status of database, models, and security middleware |

---

## Running Unit Tests
```bash
pytest tests/test_traffic_signal.py tests/test_api.py tests/test_api_sih.py -v
```
All 18 comprehensive tests pass out of the box with zero errors.

---

## Project Structure
```
backend/app/
  api/routes.py            # REST endpoints including Webster signals & repair rankings
  api/schemas.py           # Pydantic validation models
  core/security.py         # OWASP security headers, rate limiting, and input sanitization
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
tests/                     # Full automated pytest test suite
```
