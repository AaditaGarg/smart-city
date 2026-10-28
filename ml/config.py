from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path | None = None) -> dict:
    p = Path(path) if path else ROOT / "configs" / "project.yaml"
    return yaml.safe_load(p.read_text())
