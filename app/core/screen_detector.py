"""Детектирование экрана ноутбука на изображении."""

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from numpy.typing import NDArray

from config import config
from utils.geometry import BoundingBox, Polygon, order_points
from utils.logger import get_logger

logger = get_logger()


class ScreenDetectionResult:
    """Результат детектирования экрана."""
    
    def __init__(
        self,
        bounding_box: BoundingBox,
        corners: Optional[Polygon] = None,
        confidence: float = 0.0,
    ):
        self.bounding_box = bounding_box
        self.corners = corners
        self.confidence = confidence
    
    def has_corners(self) -> bool:
        """Проверить, есть ли угловые точки."""
        return self.corners is not None and len(self.corners.points) == 4


class ScreenDetector:
    """Класс для детектирования экрана ноутбука."""
    
    def __init__(self, model_path: Optional[Path] = None):
        """
        Инициализация детектора.
        
        Args:
            model_path: Путь к модели YOLO (опционально)
        """
        self.model_path = model_path
        self.model = None
        
        if model_path and model_path.exists():
            self._load_model(model_path)
    
    def _load_model(self, model_path: Path) -> None:
        """
        Загрузить модель YOLO.
        
        Args:
            model_path: Путь к модели
        """
        try:
            from ultralytics import YOLO
            logger.info(f"Loading YOLO model from {model_path}")
            self.model = YOLO(str(model_path))
            logger.info("Model loaded successfully")
        except ImportError:
            logger.warning("ultralytics not installed, ML detection disabled")
            self.model = None
        except Exception as e:
            logger.exception(f"Error loading model: {e}")
            self.model = None
    
    def detect(
        self,
        image: NDArray[np.uint8],
        use_ml: bool = True
    ) -> Optional[ScreenDetectionResult]:
        """
        Детектировать экран на изображении.
        
        Args:
            image: Входное изображение в формате BGR
            use_ml: Использовать ML модель если доступна
            
        Returns:
            Результат детектирования или None
        """
        if use_ml and self.model is not None:
            result = self._detect_with_ml(image)
            if result:
                return result
    
    def _detect_with_ml(self, image: NDArray[np.uint8]) -> Optional[ScreenDetectionResult]:
        """
        Детектирование с использованием ML модели.
        
        Args:
            image: Входное изображение
            
        Returns:
            Результат детектирования
        """
        try:
            results = self.model.predict(
                image,
                conf=config.YOLO_CONFIDENCE_THRESHOLD,
                iou=config.YOLO_IOU_THRESHOLD,
                verbose=False,
                task='segment'
            )
            
            if not results or len(results[0].boxes) == 0:
                logger.warning("No screens detected by ML model")
                return None
            
            # Детекция с максимальной уверенностью
            boxes = results[0].boxes
            best_idx = boxes.conf.argmax()
            box = boxes.xyxy[best_idx].cpu().numpy()
            confidence = float(boxes.conf[best_idx])
            
            bbox = BoundingBox(box[0], box[1], box[2], box[3])
            
            # Полигон из маски сегментации
            corners = None
            if hasattr(results[0], 'masks') and results[0].masks is not None:
                corners = self._extract_corners_from_mask(results[0].masks[best_idx])
            
            logger.info(f"ML detection: confidence={confidence:.3f}, bbox={bbox}")
            
            return ScreenDetectionResult(bbox, corners, confidence)
            
        except Exception as e:
            logger.exception(f"Error in ML detection: {e}")
            return None
    
    def _extract_corners_from_mask(self, mask) -> Optional[Polygon]:
        """
        Извлечь угловые точки из маски сегментации.
        
        Args:
            mask: Маска сегментации
            
        Returns:
            Polygon с 4 угловыми точками
        """
        try:
            #  Маска в numpy array
            if hasattr(mask, 'data'):
                mask_np = mask.data.cpu().numpy()
            else:
                mask_np = mask.cpu().numpy()
            
            # Лишние размерности
            while mask_np.ndim > 2:
                mask_np = mask_np.squeeze()
            
            # Масштаб до размера изображения
            h, w = mask_np.shape
            
            # Конвертация в uint8
            mask_np = (mask_np * 255).astype(np.uint8)
            
            logger.debug(f"Mask shape: {mask_np.shape}, min: {mask_np.min()}, max: {mask_np.max()}")
            
            # Контуры
            contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                logger.warning("No contours found in mask")
                return None
            
            logger.debug(f"Found {len(contours)} contours")
            
            # Самый большой контур
            contour = max(contours, key=cv2.contourArea)
            
            # Контур до прямоугольника
            # Вариант 1: Аппроксимация полигона
            peri = cv2.arcLength(contour, True)
            epsilon = 0.02 * peri
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            logger.debug(f"Approx has {len(approx)} points")
            
            # 4 точки - отлично
            if len(approx) == 4:
                pts = approx.reshape(4, 2).astype(np.float32)
            else:
                # Вариант 2: minAreaRect (всегда 4 точки)
                rect = cv2.minAreaRect(contour)
                pts = cv2.boxPoints(rect).astype(np.float32)
                logger.debug("Used minAreaRect fallback")
            
            # Упорядочивание точек
            pts = order_points(pts)
            
            logger.info(f"Extracted polygon with 4 corners from mask")
            return Polygon(pts)
            
        except Exception as e:
            logger.exception(f"Could not extract corners from mask: {e}")
            return None
    
    def visualize_detection(
        self,
        image: NDArray[np.uint8],
        result: ScreenDetectionResult
    ) -> NDArray[np.uint8]:
        """
        Визуализировать результат детектирования.
        
        Args:
            image: Исходное изображение
            result: Результат детектирования
            
        Returns:
            Изображение с нарисованным детектированием
        """
        vis_image = image.copy()
        
        bbox = result.bounding_box
        cv2.rectangle(
            vis_image,
            (int(bbox.x1), int(bbox.y1)),
            (int(bbox.x2), int(bbox.y2)),
            (0, 255, 0),
            2
        )
        
        # Отрисовка углов
        if result.has_corners():
            pts = result.corners.to_array().astype(np.int32)
            cv2.polylines(vis_image, [pts], True, (0, 0, 255), 2)
            
            # Отрисовка точек
            for i, pt in enumerate(pts):
                cv2.circle(vis_image, tuple(pt), 5, (255, 0, 0), -1)
                cv2.putText(
                    vis_image,
                    str(i),
                    tuple(pt + 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1
                )
        
        # Текст с уверенностью
        if result.confidence > 0:
            text = f"Confidence: {result.confidence:.2f}"
            cv2.putText(
                vis_image,
                text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )
        
        return vis_image