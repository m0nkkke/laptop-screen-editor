"""
Загрузка и конвертация форматов изображений
"""
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image
from loguru import logger

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    HEIF_SUPPORT = True
except ImportError:
    HEIF_SUPPORT = False
    logger.warning("pillow-heif не установлен, HEIF/AVIF форматы не поддерживаются")

try:
    import pillow_avif
except ImportError:
    pass

from app.config import settings


class ImageLoader:
    """Класс для загрузки и конвертации изображений"""
    
    @staticmethod
    def load_image(file_path: Path) -> Optional[np.ndarray]:
        """
        Загрузка изображения в формате numpy array (RGB)
        
        Args:
            file_path: Путь к файлу
        
        Returns:
            Изображение в формате numpy array или None при ошибке
        """
        try:
            logger.debug(f"Загрузка изображения: {file_path}")
            
            # Проверка существования файла
            if not file_path.exists():
                logger.error(f"Файл не существует: {file_path}")
                return None
            
            # Открытие изображения с помощью Pillow
            with Image.open(file_path) as img:
                # Конвертация в RGB
                if img.mode != 'RGB':
                    logger.debug(f"Конвертация из {img.mode} в RGB")
                    img = img.convert('RGB')
                
                img_array = np.array(img)
                
                logger.info(f"Изображение загружено: {file_path.name}, размер: {img_array.shape}")
                return img_array
        
        except Exception as e:
            logger.error(f"Ошибка при загрузке изображения {file_path}: {e}")
            return None
    
    @staticmethod
    def load_image_pil(file_path: Path) -> Optional[Image.Image]:
        """
        Загрузка изображения как PIL Image
        
        Args:
            file_path: Путь к файлу
        
        Returns:
            PIL Image или None при ошибке
        """
        try:
            logger.debug(f"Загрузка изображения PIL: {file_path}")
            
            if not file_path.exists():
                logger.error(f"Файл не существует: {file_path}")
                return None
            
            img = Image.open(file_path)
            
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            logger.info(f"PIL изображение загружено: {file_path.name}")
            return img
        
        except Exception as e:
            logger.error(f"Ошибка при загрузке PIL изображения {file_path}: {e}")
            return None
    
    @staticmethod
    def save_image(
        image: np.ndarray,
        output_path: Path,
        format: str = None,
        quality: int = None
    ) -> bool:
        """
        Сохранение изображения
        
        Args:
            image: Изображение в формате numpy array (RGB)
            output_path: Путь для сохранения
            format: Формат файла (jpg/png), по умолчанию из расширения
            quality: Качество (для JPEG)
        
        Returns:
            True если успешно сохранено
        """
        try:
            logger.debug(f"Сохранение изображения: {output_path}")
            
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if format is None:
                format = output_path.suffix.lower().lstrip('.')
            
            if format not in settings.SUPPORTED_OUTPUT_FORMATS:
                logger.error(f"Неподдерживаемый формат: {format}")
                return False
            
            if isinstance(image, np.ndarray):
                img = Image.fromarray(image.astype(np.uint8))
            else:
                img = image
            
            save_kwargs = {}
            
            if format == 'jpg' or format == 'jpeg':
                save_kwargs['quality'] = quality or settings.JPEG_QUALITY
                save_kwargs['optimize'] = True
            elif format == 'png':
                save_kwargs['compress_level'] = settings.PNG_COMPRESSION
                save_kwargs['optimize'] = True
            
            # Сохранение
            pil_format = format.lower()
            if pil_format == 'jpg':
                pil_format = 'jpeg'

            img.save(output_path, format=pil_format.upper(), **save_kwargs)
            
            logger.info(f"Изображение сохранено: {output_path}")
            return True
        
        except Exception as e:
            logger.error(f"Ошибка при сохранении изображения {output_path}: {e}")
            return False
    
    @staticmethod
    def get_image_info(file_path: Path) -> Optional[dict]:
        """
        Получение информации об изображении
        
        Args:
            file_path: Путь к файлу
        
        Returns:
            Словарь с информацией или None
        """
        try:
            with Image.open(file_path) as img:
                info = {
                    'width': img.width,
                    'height': img.height,
                    'mode': img.mode,
                    'format': img.format,
                    'size_mb': file_path.stat().st_size / (1024 * 1024)
                }
                return info
        except Exception as e:
            logger.error(f"Ошибка получения информации о {file_path}: {e}")
            return None
    
    @staticmethod
    def resize_for_preview(
        image: np.ndarray,
        max_size: Tuple[int, int] = None
    ) -> np.ndarray:
        """
        Изменение размера изображения для предпросмотра
        
        Args:
            image: Изображение
            max_size: Максимальный размер (ширина, высота)
        
        Returns:
            Изменённое изображение
        """
        if max_size is None:
            max_size = settings.MAX_PREVIEW_SIZE
        
        h, w = image.shape[:2]
        max_w, max_h = max_size
        
        # Если изображение меньше максимального размера
        if w <= max_w and h <= max_h:
            return image
        
        # Вычисление масштаба
        scale = min(max_w / w, max_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Изменение размера
        img_pil = Image.fromarray(image)
        img_pil = img_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        return np.array(img_pil)
    
    @staticmethod
    def convert_format(
        input_path: Path,
        output_path: Path,
        output_format: str
    ) -> bool:
        """
        Конвертация формата изображения
        
        Args:
            input_path: Входной файл
            output_path: Выходной файл
            output_format: Формат (jpg/png)
        
        Returns:
            True если успешно
        """
        try:
            logger.info(f"Конвертация {input_path} -> {output_path} ({output_format})")
            
            img = ImageLoader.load_image(input_path)
            if img is None:
                return False
            
            return ImageLoader.save_image(img, output_path, format=output_format)
        
        except Exception as e:
            logger.error(f"Ошибка конвертации {input_path}: {e}")
            return False