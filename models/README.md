# Model weights

Weights are **not stored in git** (`*.pt` / `*.onnx` are ignored). The API expects these files in this folder:

| File | Task | Classes |
|---|---|---|
| `vehicle_best.pt` (or `vehicle_best.onnx`) | vehicle detection (KITTI) | car, van, truck, tram |
| `road_damage_best.pt` (or `road_damage_best.onnx`) | road damage detection (RDD2022 India) | longitudinal_crack, transverse_crack, alligator_crack, pothole |
| `congestion_model.joblib` | congestion index (committed) | - |

## Getting them

1. **Download:** attach your trained files to a GitHub Release (Releases -> Draft a new release -> drag the files in),
   then put the release URL here and run, for example:
   `curl -L -o models/vehicle_best.pt <release-asset-url>`
2. **Or train them yourself:**
   ```bash
   python scripts/download_datasets.py
   python -m ml.training.train_yolo --task vehicle
   python -m ml.training.train_yolo --task road_damage
   ```

## ONNX note
For ONNX files the class names are taken from `configs/project.yaml` (same order used in training).

## Without weights
`/predict/*` returns **HTTP 503**. For demos only, set `ALLOW_HEURISTIC_FALLBACK=true`; the API then uses a classical-CV
heuristic and reports the model name as `<task>_heuristic_DEMO`. It is not a trained model and its accuracy is unmeasured.
