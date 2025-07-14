from pydantic_settings import BaseSettings
from pathlib import Path

# Absolute path to the model
project_root = Path(__file__).parent.parent
model_path: str = str(project_root / "backend" / "models" / "asl_model.h5")

class Settings(BaseSettings):
    app_name: str = "ASL Detector API"
    debug: bool = False
    cors_origins: list[str] = ["*"]
    model_path: str = model_path

    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()