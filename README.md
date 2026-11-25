# Smart City Intelligence & Dynamic Traffic System (SIH 2025)

**The problem:** fixed-timer signals give green to a road with 2 vehicles while a road with 100 waits at red. Fixing one junction alone just pushes the jam to the next one.

**Our solution:** camera photos or video → YOLO vehicle detection (Indian classes) → queue in PCU (IRC:106) → **Max-Pressure network signal control**, which coordinates every junction. The network is the 4 real signals on **Vikas Marg, Delhi** (positions from OpenStreetMap). A fixed safety layer sits on top of the controller, and the results are measured in a simulator against today's fixed timers.

The project also includes single-junction Webster optimization, road-damage detection and repair prioritisation (RDD2022 India), a FastAPI backend with security hardening (rate limiting, CSP/HSTS headers, magic-byte upload validation, optional API-key auth), and a React dashboard.

---

## How the signal control works

| Step | What happens |
|---|---|
| 1. Count | YOLO (ONNX Runtime or PyTorch) counts bikes, cars, autos, buses, trucks and ambulances on each approach (photo, or video at 1 frame/s). |
| 2. Queue | Counts are converted to PCU (bike 0.5, car 1.0, auto 1.2, bus/truck 2.2). If the queue fills the whole camera view, the road is marked "beyond camera" and a running balance (arrivals − departures) is kept. |
| 3. Choose | **Max-Pressure**: pressure = queue waiting − ½ × queue on the road it feeds. A road whose exit is ≥ 85 % full gets zero pressure, so we never push cars into a jam. Roads waiting long get an ageing boost. |
| 4. Time | Green lasts long enough to clear the queue (2 s start-up + 1.3 × queue ÷ saturation flow, with saturation flow S = 525 × road width PCU/h), between 10 and 60 s. It ends early if the stop line is empty in **2 checks in a row**. It is extended if the queue is still the heaviest. |
| 5. Safety | Always yellow (3–4 s) + all-red (2 s). Minimum green 10 s, maximum green 60 s. No road with waiting vehicles stays red over ~90 s, and pedestrians walk within 2 min of pressing the button. A stale camera means fixed-time fallback for that junction. |
| 6. Ambulance | Green corridor only after 2 camera frames with confidence ≥ 0.6, or a GPS/RFID request. Colour alone never triggers it. Junctions ahead turn green ~12 s before the ambulance arrives. |
| 7. Over capacity | When two roads both have more queue than a full green can clear, they take equal turns. An alert suggests diversion, and upstream junctions hold traffic back (gating). |

An **RL agent** (DQN, shared by all junctions) is trained in the same simulator and goes through the same safety layer. It is deployed only if it beats Max-Pressure on held-out random days; see `models/rl_eval.json`.

## Results

These are simulation results, not field measurements. Run `python scripts/make_network_report.py` to regenerate them. Full tables, sensitivity tests and charts are in [reports/NETWORK_RESULTS.md](reports/NETWORK_RESULTS.md).

Average wait per vehicle, mean of 5 simulated days. Same traffic for every controller, and the cameras miss some vehicles.

| Scenario | Fixed timer | Webster (1 junction) | **Max-Pressure** | AI / RL |
|---|---|---|---|---|
| Balanced traffic | 109 s | 143 s | **60 s** (−45 %) | 68 s |
| Morning rush | 155 s | 306 s | **88 s** (−43 %) | 98 s |
| Camera at J2 fails 5 min | 122 s | 189 s | **74 s** (−39 %) | 80 s |
| 2 vs 100 | 93 s | 98 s | **70 s** (−25 %) | 144 s |
| 2×2 grid rush | 186 s | 224 s | **172 s** (−8 %) | 275 s |
| 500 vs 499 (over capacity) | 512 s | 587 s | **485 s** (−5 %) | 563 s |

- **Ambulance through all 4 signals:** 8.3 min with the fixed timer, 4.8 min with Max-Pressure, and 3.7 min with Max-Pressure plus the green corridor.
- **Fairness caveat:** in the over-capacity case the side road waits longer than under the fixed plan (worst road 635 s vs 379 s). Timing cannot fix demand above capacity.
- **RL:** the agent did not beat Max-Pressure on held-out days (109 s vs 77 s), so it is not deployed.

## Quickstart

```bash
py -3.13 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m uvicorn backend.app.main:app --port 8000     # API docs: http://localhost:8000/docs
```

> **Model weights:** Put `vehicle_best.pt` (or `indian_vehicle_best.pt` / `yolov8n.pt`) and `road_damage_best.pt` in `models/` (see [`models/README.md`](models/README.md)). Without weights, prediction endpoints return **HTTP 503** (or set `ALLOW_HEURISTIC_FALLBACK=true` in `.env` for demo mode).
```bash
cd frontend && npm install && npm run dev                            # dashboard: http://localhost:5173
```


```bash
.venv\Scripts\python -m pytest tests -q
.venv\Scripts\python scripts/make_network_report.py                 # measure all controllers and scenarios
.venv\Scripts\python ml/training/train_rl.py --steps 100000          # train and evaluate the RL agent
```

Python 3.10–3.13 is supported; the venv above uses 3.13.

## Dashboard (tabs follow the presentation order)

1. **Live junction**: upload one photo per road (or a short video) and see the YOLO boxes, PCU, the pressure of each phase and a plain-language "why this decision".
2. **City network**: animated Vikas Marg corridor or 2×2 grid. Switch between Fixed / Webster / Max-Pressure / AI and pick a scenario: rush hour, 2 vs 100, 500 vs 499, ambulance, camera failure.
3. **Before vs after**: the fixed timer and the smart controller run side by side on identical traffic, with measured differences.
4. **Green corridor**: an ambulance with and without the corridor, plus a GPS request button.
5. **Model and data**: detector accuracy per class, data sources and licences, and the limits.

## Main API endpoints

| Endpoint | Purpose |
|---|---|
| `GET /network/scenarios` | Scenarios, controllers and the Vikas Marg network |
| `POST /network/simulate` | Run a controller on a scenario → timeline, metrics, alerts |
| `POST /network/compare` | Mean ± sd of metrics over several seeds |
| `POST /network/decide` / `decide-photos` / `decide-video` | Camera counts → next phase and green time, explained |
| `POST /network/emergency` | GPS/RFID ambulance request → green times at each junction ahead |
| `POST /network/pedestrian-request` | Pedestrian push-button |
| `GET /network/health` | Camera watchdog per junction (adaptive or fallback) |
| `POST /predict/vehicles`, `/predict/road-damage` | Single-image detection |
| `POST /segments/prioritize-repairs` | Road repair ranking |

## Project structure

```
decision_engine/
  network_model.py     # junctions, roads, phases, left-hand-traffic conflict check
  network_control.py   # Max-Pressure, Webster, fixed, RL controllers + safety layer + green corridor
  queue_estimator.py   # beyond-camera queues, turn-ratio learning, 2-frame ambulance confirmation
  network_sim.py       # seeded simulator of Indian mixed traffic + metrics
  rl_env.py            # Gymnasium environment for the shared RL policy
  tracking.py          # multi-object vehicle tracking across frames
  traffic_signal.py    # PCU (IRC:106), single-junction Webster + analytical delay estimate
configs/network_corridor.yaml   # Vikas Marg (OSM) + grid + scenarios
backend/app/
  api/routes.py                 # REST endpoints including Webster signals & repair rankings
  api/network_routes.py         # Multi-junction Max-Pressure, simulation & camera endpoints
  core/security.py              # Security headers, rate limiting, magic-byte upload validation, API-key
  services/model_service.py     # ONNX / PyTorch YOLO loading, 503 when weights are missing
frontend/src/tabs/              # the dashboard tabs
scripts/make_network_report.py  # measured comparison -> reports/
ml/training/train_rl.py         # RL training + deployment check
tests/                          # automated pytest test suite (CI: .github/workflows/ci.yml)
```

## Honest limits

- Results come from simulation on the real corridor geometry with typical peak-hour volumes, not from field counts.
- Connecting to real signal controllers needs the vendor protocol and approval from Delhi Traffic Police (`SignalControllerAdapter` is the plug-in point).
- When demand exceeds road capacity, timing can only share the delay; diversion and capacity are the real fixes.
- The Indian detector was trained on a 456 MB subset. The full Indian Driving Dataset (22 GB) would be stronger.
