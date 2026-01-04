"""
Перспективные преобразования экрана
"""
from typing import Optional, Tuple
import numpy as np
import cv2
from loguru import logger

from app.utils.geometry import Polygon, order_points, calculate_perspective_size


class PerspectiveTransformer:
    """Класс для перспективных преобразований"""
    
    @staticmethod
    def extract_screen_region(
        image: np.ndarray,
        polygon: Polygon
    ) -> Optional[np.ndarray]:
        """
        Извлечение области экрана с исправлением перспективы
        
        Args:
            image: Исходное изображение
            polygon: Полигон области экрана
        
        Returns:
            Извлечённая область или None
        """
        try:
            if len(polygon.points) < 4:
                logger.error(f"Недостаточно точек в полигоне: {len(polygon.points)}")
                return None
            
            logger.debug("Извлечение области экрана с исправлением перспективы")
            
            # Первые 4 точки
            points = polygon.to_numpy()[:4]
            
            # Упорядочивание точек
            ordered = order_points(points)
            
            # Вычисление размера выходного изображения
            width, height = calculate_perspective_size(ordered)
            
            # Целевые точки (прямоугольник)
            dst_points = np.array([
                [0, 0],
                [width - 1, 0],
                [width - 1, height - 1],
                [0, height - 1]
            ], dtype=np.float32)
            
            # Вычисление матрицы перспективы
            matrix = cv2.getPerspectiveTransform(ordered, dst_points)
            
            # Применение трансформации
            warped = cv2.warpPerspective(image, matrix, (width, height))
            
            logger.info(f"Область экрана извлечена: {width}x{height}")
            return warped
        
        except Exception as e:
            logger.error(f"Ошибка извлечения области экрана: {e}")
            return None
    
    @staticmethod
    def warp_image_to_polygon(
        image: np.ndarray,
        polygon: Polygon,
        canvas_shape: Tuple[int, int]
    ) -> np.ndarray:
        """
        Применение перспективы к изображению для подгонки под полигон
        
        Args:
            image: Изображение для трансформации
            polygon: Целевой полигон
            canvas_shape: Размер холста (height, width)
        
        Returns:
            Трансформированное изображение
        """
        try:
            if len(polygon.points) < 4:
                logger.error("Недостаточно точек в полигоне")
                return np.zeros((*canvas_shape, 3), dtype=np.uint8)
            
            logger.debug("Применение перспективной трансформации")
            
            # Исходные точки (углы изображения)
            h, w = image.shape[:2]
            src_points = np.array([
                [0, 0],
                [w - 1, 0],
                [w - 1, h - 1],
                [0, h - 1]
            ], dtype=np.float32)
            
            # Целевые точки (полигон)
            points = polygon.to_numpy()[:4]
            dst_points = order_points(points)
            
            # Вычисление матрицы перспективы
            matrix = cv2.getPerspectiveTransform(src_points, dst_points)
            
            # Применение трансформации
            warped = cv2.warpPerspective(
                image,
                matrix,
                (canvas_shape[1], canvas_shape[0])
            )
            
            logger.info("Перспективная трансформация применена")
            return warped
        
        except Exception as e:
            logger.error(f"Ошибка применения перспективы: {e}")
            return np.zeros((*canvas_shape, 3), dtype=np.uint8)
    
    @staticmethod
    def create_perspective_mask(
        polygon: Polygon,
        image_shape: Tuple[int, int]
    ) -> np.ndarray:
        """
        Создание маски для области с перспективой
        
        Args:
            polygon: Полигон
            image_shape: Размер изображения (height, width)
        
        Returns:
            Бинарная маска
        """
        try:
            mask = np.zeros(image_shape, dtype=np.uint8)
            
            if len(polygon.points) < 3:
                return mask
            
            points = polygon.to_int_numpy()
            
            if len(polygon.points) == 4:
                # Для четырёхугольника fillConvexPoly
                cv2.fillConvexPoly(mask, points, 255)
            else:
                # Для произвольного полигона
                cv2.fillPoly(mask, [points], 255)
            
            return mask
        
        except Exception as e:
            logger.error(f"Ошибка создания маски: {e}")
            return np.zeros(image_shape, dtype=np.uint8)
    
    @staticmethod
    def correct_perspective_distortion(
        image: np.ndarray,
        polygon: Polygon,
        target_aspect_ratio: Optional[float] = None
    ) -> Optional[np.ndarray]:
        """
        Исправление перспективных искажений
        
        Args:
            image: Исходное изображение
            polygon: Полигон с искажённой перспективой
            target_aspect_ratio: Целевое соотношение сторон
        
        Returns:
            Исправленное изображение
        """
        try:
            if len(polygon.points) < 4:
                logger.error("Недостаточно точек для исправления перспективы")
                return None
            
            # Упорядочивание точек
            points = polygon.to_numpy()[:4]
            ordered = order_points(points)
            
            # Вычисление размера
            if target_aspect_ratio is not None:
                # Заданное соотношение сторон
                width, height = calculate_perspective_size(ordered)
                current_ratio = width / height
                
                if current_ratio > target_aspect_ratio:
                    width = int(height * target_aspect_ratio)
                else:
                    height = int(width / target_aspect_ratio)
            else:
                width, height = calculate_perspective_size(ordered)
            
            # Целевые точки
            dst_points = np.array([
                [0, 0],
                [width - 1, 0],
                [width - 1, height - 1],
                [0, height - 1]
            ], dtype=np.float32)
            
            # Матрица перспективы
            matrix = cv2.getPerspectiveTransform(ordered, dst_points)
            
            # Трансформация
            corrected = cv2.warpPerspective(image, matrix, (width, height))
            
            logger.info(f"Перспектива исправлена: {width}x{height}")
            return corrected
        
        except Exception as e:
            logger.error(f"Ошибка исправления перспективы: {e}")
            return None
    
    @staticmethod
    def apply_homography(
        image: np.ndarray,
        src_points: np.ndarray,
        dst_points: np.ndarray,
        output_shape: Tuple[int, int]
    ) -> Optional[np.ndarray]:
        """
        Применение гомографии между точками
        
        Args:
            image: Исходное изображение
            src_points: Исходные точки (4x2)
            dst_points: Целевые точки (4x2)
            output_shape: Размер выходного изображения (height, width)
        
        Returns:
            Трансформированное изображение
        """
        try:
            if len(src_points) != 4 or len(dst_points) != 4:
                logger.error("Требуется ровно 4 точки для гомографии")
                return None
            
            # Вычисление матрицы гомографии
            matrix, _ = cv2.findHomography(
                src_points.astype(np.float32),
                dst_points.astype(np.float32),
                cv2.RANSAC
            )
            
            if matrix is None:
                logger.error("Не удалось вычислить матрицу гомографии")
                return None
            
            # Применение трансформации
            warped = cv2.warpPerspective(
                image,
                matrix,
                (output_shape[1], output_shape[0])
            )
            
            return warped
        
        except Exception as e:
            logger.error(f"Ошибка применения гомографии: {e}")
            return None