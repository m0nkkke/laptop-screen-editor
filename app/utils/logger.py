"""
Настройка системы логирования
"""
import sys
from pathlib import Path
from loguru import logger
from app.config import settings


def setup_logger():
    """Настройка логгера приложения"""
    logger.remove()

    # Папка для логов рядом с exe или в cwd
    if getattr(sys, "_MEIPASS", False):
        # PyInstaller создает временную папку
        base_path = Path(sys._MEIPASS)
    else:
        base_path = Path.cwd()
    
    log_dir = base_path / "logs"
    log_dir.mkdir(exist_ok=True)

    # Файловый лог
    log_file_path = log_dir / settings.LOG_FILE.name
    logger.add(
        log_file_path,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level=settings.LOG_LEVEL,
        rotation=settings.LOG_ROTATION,
        retention=settings.LOG_RETENTION,
        compression="zip",
        encoding="utf-8",
        enqueue=True,
    )

    # Консольный вывод
    if sys.stdout:
        logger.add(
            sys.stdout,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | "
                   "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            level=settings.LOG_LEVEL,
            colorize=True,
        )

    logger.info(f"Логирование инициализировано. Логи сохраняются в {log_file_path}")
    return logger


# Инициализация при импорте
logger = setup_logger()
