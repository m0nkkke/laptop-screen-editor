"""
Кадрирование изображений
"""
from typing import Optional, Tuple
import numpy as np
import cv2
from loguru import logger


class ImageCropper:
    """Класс для кадрирования изображений"""
    
    @staticmethod
    def auto_crop(
        image: np.ndarray,
        background_color: Tuple[int, int, int] = (255, 255, 255),
        margin: int = 0
    ) -> np.ndarray:
        """
        Автоматическое кадрирование изображения по содержимому
        
        Args:
            image: Исходное изображение
            background_color: Цвет фона для удаления
            margin: Отступ от границ содержимого
        
        Returns:
            Обрезанное изображение
        """
        try:
            logger.debug("Автоматическое кадрирование изображения")
            
            # Создание маски для фона
            tolerance = 30
            lower = np.array([max(0, c - tolerance) for c in background_color])
            upper = np.array([min(255, c + tolerance) for c in background_color])
            
            mask = cv2.inRange(image, lower, upper)
            
            # Инвертирование маски (объект становится белым)
            mask = cv2.bitwise_not(mask)
            
            # Поиск контуров объекта
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                logger.warning("Контуры не найдены, возврат исходного изображения")
                return image
            
            # Получение bounding box всех контуров
            x_min, y_min = image.shape[1], image.shape[0]
            x_max, y_max = 0, 0
            
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x + w)
                y_max = max(y_max, y + h)
            
            # Применение отступов
            x_min = max(0, x_min - margin)
            y_min = max(0, y_min - margin)
            x_max = min(image.shape[1], x_max + margin)
            y_max = min(image.shape[0], y_max + margin)
            
            # Кадрирование
            cropped = image[y_min:y_max, x_min:x_max]
            
            logger.info(f"Изображение обрезано: {image.shape[:2]} -> {cropped.shape[:2]}")
            return cropped
        
        except Exception as e:
            logger.error(f"Ошибка при автоматическом кадрировании: {e}")
            return image
    
    @staticmethod
    def crop_to_bbox(
        image: np.ndarray,
        x: int, y: int, width: int, height: int
    ) -> np.ndarray:
        """
        Кадрирование по заданному bounding box
        
        Args:
            image: Исходное изображение
            x, y: Координаты верхнего левого угла
            width, height: Размеры области
        
        Returns:
            Обрезанное изображение
        """
        try:
            # Проверка границ
            h, w = image.shape[:2]
            x = max(0, min(x, w))
            y = max(0, min(y, h))
            x2 = max(0, min(x + width, w))
            y2 = max(0, min(y + height, h))
            
            cropped = image[y:y2, x:x2]
            logger.debug(f"Кадрирование по bbox: ({x}, {y}, {width}, {height})")
            return cropped
        
        except Exception as e:
            logger.error(f"Ошибка при кадрировании по bbox: {e}")
            return image
    
    @staticmethod
    def crop_to_aspect_ratio(
        image: np.ndarray,
        aspect_ratio: float,
        from_center: bool = True
    ) -> np.ndarray:
        """
        Кадрирование до заданного соотношения сторон
        
        Args:
            image: Исходное изображение
            aspect_ratio: Соотношение сторон (ширина/высота)
            from_center: Кадрировать от центра
        
        Returns:
            Обрезанное изображение
        """
        try:
            h, w = image.shape[:2]
            current_ratio = w / h
            
            if abs(current_ratio - aspect_ratio) < 0.01:
                return image
            
            if current_ratio > aspect_ratio:
                # Изображение шире, обрезаем по ширине
                new_width = int(h * aspect_ratio)
                if from_center:
                    x_start = (w - new_width) // 2
                else:
                    x_start = 0
                cropped = image[:, x_start:x_start + new_width]
            else:
                # Изображение выше, обрезаем по высоте
                new_height = int(w / aspect_ratio)
                if from_center:
                    y_start = (h - new_height) // 2
                else:
                    y_start = 0
                cropped = image[y_start:y_start + new_height, :]
            
            logger.info(f"Кадрирование до соотношения {aspect_ratio:.2f}")
            return cropped
        
        except Exception as e:
            logger.error(f"Ошибка при кадрировании до соотношения сторон: {e}")
            return image
    
    @staticmethod
    def smart_crop(
        image: np.ndarray,
        target_width: int,
        target_height: int
    ) -> np.ndarray:
        """
        Умное кадрирование с сохранением важного содержимого
        
        Args:
            image: Исходное изображение
            target_width: Целевая ширина
            target_height: Целевая высота
        
        Returns:
            Обрезанное изображение
        """
        try:
            h, w = image.shape[:2]
            target_ratio = target_width / target_height
            current_ratio = w / h
            
            # Если размеры уже подходящие
            if w == target_width and h == target_height:
                return image
            
            # Сначала кадрируем до нужного соотношения сторон
            if abs(current_ratio - target_ratio) > 0.01:
                image = ImageCropper.crop_to_aspect_ratio(image, target_ratio, from_center=True)
            
            # Затем масштабируем до нужного размера
            from app.core.resizer import ImageResizer
            image = ImageResizer.resize(image, target_width, target_height, maintain_aspect=False)
            
            return image
        
        except Exception as e:
            logger.error(f"Ошибка при умном кадрировании: {e}")
            return image
    
    @staticmethod
    def pad_to_aspect_ratio(
        image: np.ndarray,
        aspect_ratio: float,
        padding_color: Tuple[int, int, int] = (255, 255, 255)
    ) -> np.ndarray:
        """
        Добавление padding для достижения нужного соотношения сторон
        
        Args:
            image: Исходное изображение
            aspect_ratio: Целевое соотношение сторон
            padding_color: Цвет padding
        
        Returns:
            Изображение с padding
        """
        try:
            h, w = image.shape[:2]
            current_ratio = w / h
            
            if abs(current_ratio - aspect_ratio) < 0.01:
                return image
            
            if current_ratio < aspect_ratio:
                new_width = int(h * aspect_ratio)
                pad_total = new_width - w
                pad_left = pad_total // 2
                pad_right = pad_total - pad_left
                
                padded = cv2.copyMakeBorder(
                    image,
                    0, 0, pad_left, pad_right,
                    cv2.BORDER_CONSTANT,
                    value=padding_color
                )
            else:
                new_height = int(w / aspect_ratio)
                pad_total = new_height - h
                pad_top = pad_total // 2
                pad_bottom = pad_total - pad_top
                
                padded = cv2.copyMakeBorder(
                    image,
                    pad_top, pad_bottom, 0, 0,
                    cv2.BORDER_CONSTANT,
                    value=padding_color
                )
            
            logger.info(f"Добавлен padding для соотношения {aspect_ratio:.2f}")
            return padded
        
        except Exception as e:
            logger.error(f"Ошибка при добавлении padding: {e}")
            return image