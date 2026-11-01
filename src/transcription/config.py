from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="TRANSCRIPTION_", env_file=".env", extra="ignore")
    model: str = "medium"
    batch_size: int = Field(default=1, ge=1)
    separation: bool = False
    temp_dir: Path | None = None
    max_bytes: int = Field(default=100 * 1024 * 1024, gt=0)
    max_duration: float = Field(default=1800, gt=0)
    timeout: float = Field(default=3600, gt=0)
