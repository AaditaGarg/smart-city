from pathlib import Path

from ..core.config import settings
from ..db.session import get_db  # noqa: F401  (re-export)
from ..services.model_service import ModelRegistry

ROOT = Path(__file__).resolve().parents[3]
_registry = None


def get_registry() -> ModelRegistry:
    global _registry
    if _registry is None:
        _registry = ModelRegistry(ROOT / settings.model_dir)
    return _registry
