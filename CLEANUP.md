# Run these once in your local clone (these are deletions, which cannot be uploaded as files)

```bash
git rm --cached smartcity.db          # committed database; .gitignore now excludes *.db
git rm -r ml_models/mobilenet         # leftover deploy.prototxt, unused
git add -A
git commit -m "Fix: 503 when weights missing, security hardening, honest README, CI, tests"
git push
```

Then create a GitHub Release and attach `vehicle_best.pt`, `road_damage_best.pt` (and optionally the `.onnx` exports),
and paste its URL into `models/README.md`.
