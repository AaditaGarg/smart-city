from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

_ROOT = Path(__file__).resolve().parents[3]
_SQLITE_PATH = _ROOT / "smartcity.db"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    database_url: str = f"sqlite:///{_SQLITE_PATH}"
    model_dir: str = "models"
    cors_origins: str = "http://localhost:5173,http://localhost:3000,http://127.0.0.1:5173"
    api_key: str = "smartcity-secure-key-2025"
    require_api_key: bool = False   # set True to strictly enforce X-API-Key on mutations
    rate_limit_per_minute: int = 240
    max_upload_mb: int = 15


settings = Settings()
