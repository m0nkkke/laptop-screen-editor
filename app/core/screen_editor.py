"""
Редактирование области экрана: заливка цветом или замена изображением
"""
from typing import Optional, Tuple
import numpy as np
import cv2
from loguru import logger

from app.utils.geometry import Polygon, order_points


class ScreenEditor:
    """Класс для редактирования области экрана"""
    
    @staticmethod
    def fill_screen_black(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Заливка области экрана чёрным цветом
        
        Args:
            image: Исходное изображение
            mask: Маска экрана
        
        Returns:
            Изображение с залитым экраном
        """
        try:
            logger.debug("Заливка экрана чёрным цветом")
            
            # Копия изображения
            result = image.copy()
            
            # Применение маски (заливка чёрным)
            result[mask > 127] = [0, 0, 0]
            
            logger.info("Экран успешно залит чёрным цветом")
            return result
        
        except Exception as e:
            logger.error(f"Ошибка при заливке экрана: {e}")
            return image
    
    @staticmethod
    def fill_screen_color(
        image: np.ndarray,
        mask: np.ndarray,
        color: Tuple[int, int, int]
    ) -> np.ndarray:
        """
        Заливка области экрана указанным цветом
        
        Args:
            image: Исходное изображение
            mask: Маска экрана
            color: Цвет в формате RGB
        
        Returns:
            Изображение с залитым экраном
        """
        try:
            logger.debug(f"Заливка экрана цветом: {color}")
            
            result = image.copy()
            result[mask > 127] = color
            
            logger.info("Экран успешно залит цветом")
            return result
        
        except Exception as e:
            logger.error(f"Ошибка при заливке экрана цветом: {e}")
            return image
    
    @staticmethod
    def replace_screen_with_image(
        laptop_image: np.ndarray,
        mask: np.ndarray,
        screen_content: np.ndarray,
        blend: bool = True
    ) -> np.ndarray:
        """
        Замена содержимого экрана на пользовательское изображение
        
        Args:
            laptop_image: Изображение ноутбука
            mask: Маска области экрана
            screen_content: Изображение для вставки
            blend: Применить сглаживание границ
        
        Returns:
            Изображение с заменённым экраном
        """
        try:
            logger.debug("Замена содержимого экрана")
            
            # Копия изображения
            result = laptop_image.copy()
            
            # Получение bounding box маски
            y_indices, x_indices = np.where(mask > 127)
            
            if len(y_indices) == 0 or len(x_indices) == 0:
                logger.warning("Пустая маска, возврат исходного изображения")
                return laptop_image
            
            x_min, x_max = x_indices.min(), x_indices.max()
            y_min, y_max = y_indices.min(), y_indices.max()
            
            width = x_max - x_min
            height = y_max - y_min
            
            # Изменение размера содержимого экрана
            screen_resized = cv2.resize(
                screen_content,
                (width, height),
                interpolation=cv2.INTER_LANCZOS4
            )
            
            # Извлечение маски для региона
            mask_region = mask[y_min:y_max, x_min:x_max]
            
            # Применение сглаживания
            if blend:
                # Размытие краёв маски для плавного перехода
                mask_blurred = cv2.GaussianBlur(mask_region.astype(np.float32), (5, 5), 0)
                mask_blurred = mask_blurred / 255.0
                mask_blurred = np.stack([mask_blurred] * 3, axis=-1)
                
                # Смешивание
                region = result[y_min:y_max, x_min:x_max]
                blended = (screen_resized * mask_blurred + region * (1 - mask_blurred)).astype(np.uint8)
                result[y_min:y_max, x_min:x_max] = blended
            else:
                # Простая замена
                result[y_min:y_max, x_min:x_max][mask_region > 127] = \
                    screen_resized[mask_region > 127]
            
            logger.info("Содержимое экрана успешно заменено")
            return result
        
        except Exception as e:
            logger.error(f"Ошибка при замене содержимого экрана: {e}")
            return laptop_image
    
    @staticmethod
    def replace_screen_with_perspective(
        laptop_image: np.ndarray,
        polygon: Polygon,
        screen_content: np.ndarray
    ) -> np.ndarray:
        """
        Замена содержимого экрана с учётом перспективы
        
        Args:
            laptop_image: Изображение ноутбука
            polygon: Полигон области экрана
            screen_content: Изображение для вставки
        
        Returns:
            Изображение с заменённым экраном
        """
        try:
            logger.debug("Замена содержимого экрана с перспективой")
            
            # Упорядочивание точек полигона
            if len(polygon.points) < 4:
                logger.error(f"Недостаточно точек в полигоне: {len(polygon.points)}")
                return laptop_image
            
            # 4 точки перспективы
            points = polygon.to_numpy()[:4]
            ordered_points = order_points(points)
            
            # Размеры содержимого экрана
            h, w = screen_content.shape[:2]
            
            # Целевые точки (прямоугольник)
            dst_points = np.array([
                [0, 0],
                [w - 1, 0],
                [w - 1, h - 1],
                [0, h - 1]
            ], dtype=np.float32)
            
            # Вычисление обратной матрицы перспективы
            # (от прямоугольника к перспективе)
            matrix = cv2.getPerspectiveTransform(dst_points, ordered_points)
            
            # Применение перспективной трансформации к содержимому
            warped = cv2.warpPerspective(
                screen_content,
                matrix,
                (laptop_image.shape[1], laptop_image.shape[0])
            )
            
            # Маска для наложения
            mask = np.zeros(laptop_image.shape[:2], dtype=np.uint8)
            cv2.fillConvexPoly(mask, ordered_points.astype(np.int32), 255)
            
            # Маска для плавного перехода
            mask_blurred = cv2.GaussianBlur(mask, (5, 5), 0).astype(np.float32) / 255.0
            mask_blurred = np.stack([mask_blurred] * 3, axis=-1)
            
            # Смешивание
            result = laptop_image.copy()
            result = (warped * mask_blurred + result * (1 - mask_blurred)).astype(np.uint8)
            
            logger.info("Содержимое экрана успешно заменено с перспективой")
            return result
        
        except Exception as e:
            logger.error(f"Ошибка при замене с перспективой: {e}")
            return laptop_image
    
    @staticmethod
    def create_screen_mask_from_polygon(
        image_shape: Tuple[int, int],
        polygon: Polygon
    ) -> np.ndarray:
        """
        Создание маски из полигона
        
        Args:
            image_shape: Форма изображения (height, width)
            polygon: Полигон
        
        Returns:
            Бинарная маска
        """
        try:
            mask = np.zeros(image_shape, dtype=np.uint8)
            points = polygon.to_int_numpy().reshape((-1, 1, 2))
            cv2.fillPoly(mask, [points], 255)
            return mask
        
        except Exception as e:
            logger.error(f"Ошибка создания маски из полигона: {e}")
            return np.zeros(image_shape, dtype=np.uint8)