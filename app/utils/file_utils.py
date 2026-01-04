"""
Утилиты для работы с файлами
"""
from pathlib import Path
from typing import List, Optional
from loguru import logger

from app.config import settings


def get_supported_files(directory: Path, recursive: bool = False) -> List[Path]:
    """
    Получение списка поддерживаемых файлов из директории
    
    Args:
        directory: Путь к директории
        recursive: Рекурсивный поиск
    
    Returns:
        Список путей к файлам
    """
    if not directory.exists() or not directory.is_dir():
        logger.warning(f"Директория не существует: {directory}")
        return []
    
    files = []
    
    if recursive:
        pattern = "**/*"
    else:
        pattern = "*"
    
    for file_path in directory.glob(pattern):
        if file_path.is_file() and is_supported_format(file_path):
            files.append(file_path)
    
    logger.info(f"Найдено {len(files)} поддерживаемых файлов в {directory}")
    return sorted(files)


def is_supported_format(file_path: Path) -> bool:
    """
    Проверка, является ли формат файла поддерживаемым
    
    Args:
        file_path: Путь к файлу
    
    Returns:
        True если формат поддерживается
    """
    suffix = file_path.suffix.lower().lstrip('.')
    return suffix in settings.SUPPORTED_INPUT_FORMATS


def get_output_filename(
    input_path: Path,
    output_format: str,
    suffix: str = "_processed"
) -> str:
    """
    Генерация имени выходного файла
    
    Args:
        input_path: Путь к входному файлу
        output_format: Формат выходного файла
        suffix: Суффикс для добавления к имени
    
    Returns:
        Имя выходного файла
    """
    stem = input_path.stem
    return f"{stem}{suffix}.{output_format}"


def ensure_directory(directory: Path) -> None:
    """
    Создание директории, если она не существует
    
    Args:
        directory: Путь к директории
    """
    directory.mkdir(parents=True, exist_ok=True)


def get_file_size_mb(file_path: Path) -> float:
    """
    Получение размера файла в мегабайтах
    
    Args:
        file_path: Путь к файлу
    
    Returns:
        Размер файла в MB
    """
    if not file_path.exists():
        return 0.0
    
    size_bytes = file_path.stat().st_size
    return size_bytes / (1024 * 1024)


def validate_output_directory(directory: Path) -> bool:
    """
    Проверка возможности записи в директорию
    
    Args:
        directory: Путь к директории
    
    Returns:
        True если можно записывать
    """
    try:
        ensure_directory(directory)
        
        # Пробуем создать тестовый файл
        test_file = directory / ".write_test"
        test_file.touch()
        test_file.unlink()
        
        return True
    except Exception as e:
        logger.error(f"Невозможно записать в директорию {directory}: {e}")
        return False


def safe_filename(filename: str) -> str:
    """
    Очистка имени файла от недопустимых символов
    
    Args:
        filename: Имя файла
    
    Returns:
        Безопасное имя файла
    """
    # Замена недопустимых символов
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    
    return filename


def get_unique_filename(directory: Path, filename: str) -> Path:
    """
    Получение уникального имени файла (добавление счётчика если файл существует)
    
    Args:
        directory: Директория
        filename: Имя файла
    
    Returns:
        Уникальный путь к файлу
    """
    file_path = directory / filename
    
    if not file_path.exists():
        return file_path
    
    stem = file_path.stem
    suffix = file_path.suffix
    counter = 1
    
    while True:
        new_filename = f"{stem}_{counter}{suffix}"
        new_path = directory / new_filename
        
        if not new_path.exists():
            return new_path
        
        counter += 1