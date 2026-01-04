"""
Глобальные настройки приложения
"""
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Настройки приложения"""
    
    # Пути
    APP_DIR: Path = Path(__file__).parent
    PROJECT_ROOT: Path = APP_DIR.parent
    ML_MODELS_DIR: Path = PROJECT_ROOT / "app" / "ml" / "models"
    
    # Модель ML
    MODEL_PATH: Path = ML_MODELS_DIR / "screen_detector.pt"
    MODEL_CONFIDENCE: float = Field(default=0.25, ge=0.0, le=1.0)
    MODEL_IOU_THRESHOLD: float = Field(default=0.45, ge=0.0, le=1.0)
    
    # Обработка изображений
    SUPPORTED_INPUT_FORMATS: tuple = ("jpg", "jpeg", "png", "webp", "avif", "heif")
    SUPPORTED_OUTPUT_FORMATS: tuple = ("jpg", "png")
    DEFAULT_OUTPUT_FORMAT: Literal["jpg", "png"] = "png"
    
    # Размеры изображений
    MAX_PREVIEW_SIZE: tuple[int, int] = (1200, 900)
    DEFAULT_OUTPUT_WIDTH: int = 2000
    DEFAULT_OUTPUT_HEIGHT: int = 1500
    JPEG_QUALITY: int = Field(default=95, ge=1, le=100)
    PNG_COMPRESSION: int = Field(default=6, ge=0, le=9)
    
    # UI
    WINDOW_TITLE: str = "Laptop Screen Editor"
    WINDOW_MIN_WIDTH: int = 1024
    WINDOW_MIN_HEIGHT: int = 768
    
    # Обработка
    MAX_THREADS: int = 4
    BATCH_SIZE: int = 10
    
    # Логирование
    LOG_LEVEL: str = "INFO"
    LOG_FILE: Path = PROJECT_ROOT / "logs" / "app.log"
    LOG_ROTATION: str = "10 MB"
    LOG_RETENTION: str = "1 week"
    
    class Config:
        env_prefix = "LAPTOP_EDITOR_"
        case_sensitive = False


# Глобальный экземпляр настроек
settings = Settings()

# Создание необходимых директорий
settings.LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
settings.ML_MODELS_DIR.mkdir(parents=True, exist_ok=True)