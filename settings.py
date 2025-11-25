from pydantic_settings import BaseSettings
import os
import json
from typing import Any


class ModelsSettings(BaseSettings):
    MODEL_NAME: str = "qwen2.5:7b"
    TEMPERATURE: float = 0
    BASE_URL: str = "http://localhost:11434/v1"
    API_KEY: str = "ollama"


class Settings(BaseSettings):
    models_settings: ModelsSettings


file_name = "models_settings.json"
config_data: dict[str, Any] = {}
if os.path.exists(file_name):
    with open(file_name, "r", encoding="utf-8") as f:
        key = file_name.replace(".json", "")
        config_data[key] = json.load(f)
settings = Settings.model_validate(config_data)
