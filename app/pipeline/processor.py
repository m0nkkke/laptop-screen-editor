"""
Процессор для пакетной обработки изображений
"""
from pathlib import Path
from typing import List, Optional, Callable, Dict, Any
from dataclasses import dataclass
from enum import Enum
import numpy as np
from loguru import logger

from app.config import settings
from app.core.image_loader import ImageLoader
from app.ml.inference import ScreenDetector
from app.core.screen_editor import ScreenEditor
from app.core.cropper import ImageCropper
from app.core.resizer import ImageResizer
from app.utils.file_utils import get_output_filename, ensure_directory


class ScreenFillMode(Enum):
    """Режим заливки экрана"""
    BLACK = "black"
    COLOR = "color"
    IMAGE = "image"
    NONE = "none"


@dataclass
class ProcessingOptions:
    """Опции обработки"""
    # Детектирование
    detect_confidence: float = 0.25
    
    # Режим заливки экрана
    fill_mode: ScreenFillMode = ScreenFillMode.BLACK
    fill_color: tuple = (0, 0, 0)
    fill_image_path: Optional[Path] = None
    use_perspective: bool = True
    
    # Кадрирование
    auto_crop: bool = False
    crop_margin: int = 10
    crop_aspect_ratio: Optional[float] = None
    
    # Масштабирование
    resize: bool = False
    target_width: Optional[int] = None
    target_height: Optional[int] = None
    maintain_aspect: bool = True
    
    # Сохранение
    output_format: str = "png"
    output_quality: int = 95
    output_suffix: str = "_processed"


@dataclass
class ProcessingResult:
    """Результат обработки одного изображения"""
    input_path: Path
    output_path: Optional[Path] = None
    success: bool = False
    error: Optional[str] = None
    processing_time: float = 0.0
    screen_detected: bool = False
    screen_confidence: float = 0.0


class ImageProcessor:
    """Процессор для обработки изображений"""
    
    def __init__(self, detector: Optional[ScreenDetector] = None):
        """
        Инициализация процессора
        
        Args:
            detector: Детектор экрана (если None, создаётся новый)
        """
        self.detector = detector or ScreenDetector()
        self.fill_image_cache: Optional[np.ndarray] = None
        logger.info("Процессор изображений инициализирован")
    
    def process_single(
        self,
        input_path: Path,
        output_dir: Path,
        options: ProcessingOptions
    ) -> ProcessingResult:
        """
        Обработка одного изображения
        
        Args:
            input_path: Путь к входному файлу
            output_dir: Директория для выходных файлов
            options: Опции обработки
        
        Returns:
            Результат обработки
        """
        import time
        start_time = time.time()
        
        result = ProcessingResult(input_path=input_path)
        
        try:
            logger.info(f"Обработка: {input_path.name}")
            
            # 1. Загрузка изображения
            image = ImageLoader.load_image(input_path)
            if image is None:
                result.error = "Ошибка загрузки изображения"
                return result
            
            # 2. Детектирование экрана
            detection = self.detector.detect_screen(image, options.detect_confidence)
            
            if detection is None:
                logger.warning(f"Экран не найден: {input_path.name}")
                result.screen_detected = False
                
                # Оригинал
                if options.fill_mode != ScreenFillMode.NONE:
                    result.error = "Экран не найден"
                    return result
            else:
                mask, polygon = detection
                result.screen_detected = True
                logger.info(f"Экран найден: {input_path.name}")
                
                # 3. Заливка/замена экрана
                if options.fill_mode == ScreenFillMode.BLACK:
                    image = ScreenEditor.fill_screen_black(image, mask)
                
                elif options.fill_mode == ScreenFillMode.COLOR:
                    image = ScreenEditor.fill_screen_color(image, mask, options.fill_color)
                
                elif options.fill_mode == ScreenFillMode.IMAGE:
                    if options.fill_image_path is not None:
                        # Кэширование заливочного изображения
                        if self.fill_image_cache is None:
                            self.fill_image_cache = ImageLoader.load_image(options.fill_image_path)
                        
                        if self.fill_image_cache is not None:
                            if options.use_perspective and len(polygon.points) >= 4:
                                image = ScreenEditor.replace_screen_with_perspective(
                                    image, polygon, self.fill_image_cache
                                )
                            else:
                                image = ScreenEditor.replace_screen_with_image(
                                    image, mask, self.fill_image_cache
                                )
            
            # 4. Кадрирование
            if options.auto_crop:
                image = ImageCropper.auto_crop(image, margin=options.crop_margin)
            
            if options.crop_aspect_ratio is not None:
                image = ImageCropper.crop_to_aspect_ratio(image, options.crop_aspect_ratio)
            
            # 5. Масштабирование
            if options.resize:
                image = ImageResizer.resize(
                    image,
                    options.target_width,
                    options.target_height,
                    options.maintain_aspect
                )
            
            # 6. Сохранение
            ensure_directory(output_dir)
            output_filename = get_output_filename(
                input_path,
                options.output_format,
                options.output_suffix
            )
            output_path = output_dir / output_filename
            
            success = ImageLoader.save_image(
                image,
                output_path,
                format=options.output_format,
                quality=options.output_quality
            )
            
            if success:
                result.success = True
                result.output_path = output_path
                logger.info(f"Обработка завершена: {output_path.name}")
            else:
                result.error = "Ошибка сохранения"
        
        except Exception as e:
            logger.error(f"Ошибка обработки {input_path}: {e}")
            result.error = str(e)
        
        finally:
            result.processing_time = time.time() - start_time
        
        return result
    
    def process_batch(
        self,
        input_files: List[Path],
        output_dir: Path,
        options: ProcessingOptions,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> List[ProcessingResult]:
        """
        Пакетная обработка изображений
        
        Args:
            input_files: Список входных файлов
            output_dir: Директория для выходных файлов
            options: Опции обработки
            progress_callback: Callback для отслеживания прогресса
        
        Returns:
            Список результатов обработки
        """
        logger.info(f"Начало пакетной обработки: {len(input_files)} файлов")
        
        results = []
        total = len(input_files)
        
        # Предзагрузка заливочного изображения
        if options.fill_mode == ScreenFillMode.IMAGE and options.fill_image_path:
            self.fill_image_cache = ImageLoader.load_image(options.fill_image_path)
            if self.fill_image_cache is None:
                logger.error(f"Не удалось загрузить изображение: {options.fill_image_path}")
        
        for i, input_path in enumerate(input_files, 1):
            # Callback прогресса
            if progress_callback:
                progress_callback(i, total, input_path.name)
            
            # Обработка
            result = self.process_single(input_path, output_dir, options)
            results.append(result)
        
        # Статистика
        successful = sum(1 for r in results if r.success)
        failed = total - successful
        detected = sum(1 for r in results if r.screen_detected)
        
        logger.info(f"Пакетная обработка завершена: {successful}/{total} успешно")
        logger.info(f"Экран обнаружен: {detected}/{total} файлов")
        
        if failed > 0:
            logger.warning(f"Не удалось обработать: {failed} файлов")
        
        return results
    
    def get_processing_report(self, results: List[ProcessingResult]) -> Dict[str, Any]:
        """
        Генерация отчёта об обработке
        
        Args:
            results: Результаты обработки
        
        Returns:
            Словарь с отчётом
        """
        total = len(results)
        successful = sum(1 for r in results if r.success)
        failed = total - successful
        detected = sum(1 for r in results if r.screen_detected)
        
        total_time = sum(r.processing_time for r in results)
        avg_time = total_time / total if total > 0 else 0
        
        failed_files = [r.input_path.name for r in results if not r.success]
        
        report = {
            'total_files': total,
            'successful': successful,
            'failed': failed,
            'screens_detected': detected,
            'total_time': total_time,
            'average_time': avg_time,
            'failed_files': failed_files
        }
        
        return report