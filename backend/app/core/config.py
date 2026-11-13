from pathlib import Path

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_ROOT = Path(__file__).resolve().parents[3]
_SQLITE_PATH = _ROOT / "smartcity.db"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    database_url: str = f"sqlite:///{_SQLITE_PATH}"
    model_dir: str = "models"
    cors_origins: str = "http://localhost:5173,http://localhost:3000,http://127.0.0.1:5173"

    # No default secret is shipped. Set API_KEY in .env and REQUIRE_API_KEY=true to protect POST/PATCH endpoints.
    api_key: str = ""
    require_api_key: bool = False
    rate_limit_per_minute: int = 240
    max_upload_mb: int = 15

    # Demo only: when trained weights are missing, use a classical-CV heuristic instead of returning HTTP 503.
    allow_heuristic_fallback: bool = False

    @model_validator(mode="after")
    def _check_auth(self):
        if self.require_api_key and not self.api_key:
            raise ValueError("REQUIRE_API_KEY=true needs API_KEY to be set in the environment/.env")
        return self


settings = Settings()
