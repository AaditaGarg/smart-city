from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    database_url: str = "postgresql+psycopg2://smartcity:smartcity_pw@localhost:5432/smartcity"
    model_dir: str = "models"
    cors_origins: str = "http://localhost:5173"


settings = Settings()
