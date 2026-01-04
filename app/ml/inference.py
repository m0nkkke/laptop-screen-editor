"""
Инференс модели YOLO для сегментации экрана ноутбука
"""
from pathlib import Path
from typing import Optional, Tuple, List
import numpy as np
from loguru import logger

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logger.warning("Ultralytics не установлен. ML функции недоступны.")

from app.config import settings
from app.utils.geometry import Polygon


class ScreenDetector:
    """Детектор экрана ноутбука с использованием YOLOv8 сегментации"""
    
    def __init__(self, model_path: Optional[Path] = None):
        """
        Инициализация детектора
        
        Args:
            model_path: Путь к модели YOLO
        """
        self.model_path = model_path or settings.MODEL_PATH
        self.model = None
        self.is_loaded = False
        
        # Загрузка модели при инициализации
        self.load_model()
    
    def load_model(self) -> bool:
        """
        Загрузка модели YOLO
        
        Returns:
            True если модель успешно загружена
        """
        if not YOLO_AVAILABLE:
            logger.error("Ultralytics не установлен. Установите: poetry install --with ml")
            return False
        
        try:
            if not self.model_path.exists():
                logger.error(f"Модель не найдена: {self.model_path}")
                logger.info("Поместите файл модели screen_detector.pt в app/ml/models/")
                return False
            
            logger.info(f"Загрузка модели: {self.model_path}")
            self.model = YOLO(str(self.model_path))
            self.is_loaded = True
            logger.info("Модель успешно загружена")
            return True
        
        except Exception as e:
            logger.error(f"Ошибка загрузки модели: {e}")
            self.is_loaded = False
            return False
    
    def detect_screen(
        self,
        image: np.ndarray,
        confidence: float = None
    ) -> Optional[Tuple[np.ndarray, Polygon]]:
        """
        Определение экрана на изображении
        
        Args:
            image: Изображение в формате numpy array (RGB)
            confidence: Порог уверенности
        
        Returns:
            Кортеж (маска, полигон) или None если экран не найден
        """
        if not self.is_loaded:
            logger.error("Модель не загружена")
            return None
        
        if confidence is None:
            confidence = settings.MODEL_CONFIDENCE
        
        try:
            logger.debug(f"Запуск детектирования, изображение: {image.shape}")
            
            # Инференс
            results = self.model.predict(
                image,
                conf=confidence,
                iou=settings.MODEL_IOU_THRESHOLD,
                verbose=False
            )
            
            if len(results) == 0:
                logger.warning("Результаты детектирования пусты")
                return None
            
            result = results[0]
            
            # Проверка наличия масок
            if result.masks is None or len(result.masks) == 0:
                logger.warning("Маски не найдены на изображении")
                return None
            
            # Первая маска (с наибольшей уверенностью)
            mask_data = result.masks.data[0].cpu().numpy()
            
            # Масштабирование маски до размера изображения
            if mask_data.shape != image.shape[:2]:
                from PIL import Image as PILImage
                mask_pil = PILImage.fromarray((mask_data * 255).astype(np.uint8))
                mask_pil = mask_pil.resize((image.shape[1], image.shape[0]), PILImage.Resampling.NEAREST)
                mask = np.array(mask_pil)
            else:
                mask = (mask_data * 255).astype(np.uint8)
            
            # Извлечение полигона из маски
            polygon = self._mask_to_polygon(mask)
            
            if polygon is None:
                logger.warning("Не удалось извлечь полигон из маски")
                return None
            
            confidence_score = float(result.boxes.conf[0]) if result.boxes is not None else 0.0
            logger.info(f"Экран найден, уверенность: {confidence_score:.2f}")
            
            return mask, polygon
        
        except Exception as e:
            logger.error(f"Ошибка при детектировании экрана: {e}")
            return None
    
    def _mask_to_polygon(self, mask: np.ndarray) -> Optional[Polygon]:
        """
        Извлечение полигона из маски
        
        Args:
            mask: Бинарная маска
        
        Returns:
            Полигон или None
        """
        import cv2
        
        try:
            # Бинаризация маски
            _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            
            # Поиск контуров
            contours, _ = cv2.findContours(
                binary_mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            if not contours:
                return None
            
            # Самый большой контур
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Аппроксимация полигона
            epsilon = 0.01 * cv2.arcLength(largest_contour, True)
            approx = cv2.approxPolyDP(largest_contour, epsilon, True)
            
            # Конвертация в список точек
            points = [(float(p[0][0]), float(p[0][1])) for p in approx]
            
            return Polygon(points)
        
        except Exception as e:
            logger.error(f"Ошибка извлечения полигона: {e}")
            return None
    
    def detect_batch(
        self,
        images: List[np.ndarray],
        confidence: float = None
    ) -> List[Optional[Tuple[np.ndarray, Polygon]]]:
        """
        Пакетное детектирование экранов
        
        Args:
            images: Список изображений
            confidence: Порог уверенности
        
        Returns:
            Список результатов (маска, полигон) или None для каждого изображения
        """
        results = []
        
        for i, image in enumerate(images):
            logger.debug(f"Обработка изображения {i+1}/{len(images)}")
            result = self.detect_screen(image, confidence)
            results.append(result)
        
        return results
    
    def visualize_detection(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        polygon: Optional[Polygon] = None
    ) -> np.ndarray:
        """
        Визуализация результата детектирования
        
        Args:
            image: Исходное изображение
            mask: Маска
            polygon: Полигон (опционально)
        
        Returns:
            Изображение с визуализацией
        """
        import cv2
        
        # Копия изображения
        vis_image = image.copy()
        
        # Наложение маски (полупрозрачная)
        colored_mask = np.zeros_like(image)
        colored_mask[:, :, 1] = mask  # Зелёный канал
        
        vis_image = cv2.addWeighted(vis_image, 0.7, colored_mask, 0.3, 0)
        
        # Отрисовка полигона
        if polygon is not None:
            points = polygon.to_int_numpy()
            cv2.polylines(vis_image, [points], True, (0, 255, 0), 3)
        
        return vis_image