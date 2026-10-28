# Datasets

| Dataset                                    | Task              | Size                                       | Format         | License                        | Source                                                    |
| ------------------------------------------ | ----------------- | ------------------------------------------ | -------------- | ------------------------------ | --------------------------------------------------------- |
| KITTI (Ultralytics YOLO conversion)        | Vehicle detection | 390 MB zip; 5,985 train + 1,496 val images | YOLO txt       | KITTI: non-commercial research | github.com/ultralytics/assets (kitti.zip)                 |
| RDD2022 (India; optional Czech, Japan, US) | Cracks + potholes | India ~502 MB zip                          | Pascal VOC XML | CC BY-SA 4.0                   | figshare 21431547 / github.com/sekilab/RoadDamageDetector |

Verified by inspecting the KITTI archive (counts above are measured, not quoted).
Notes: RDD2022 test splits have no labels, so Step 3 re-splits the labelled train data. A large share of RDD2022 images have no damage and are kept as negatives.
Congestion has no public ground-truth label here; it is derived from vehicle detections (documented in Step 4).
