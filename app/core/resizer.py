"""
Масштабирование изображений с сохранением пропорций
"""
from typing import Optional, Tuple
import numpy as np
import cv2
from loguru import logger


class ImageResizer:
    """Класс для масштабирования изображений"""
    
    @staticmethod
    def resize(
        image: np.ndarray,
        width: Optional[int] = None,
        height: Optional[int] = None,
        maintain_aspect: bool = True,
        interpolation: int = cv2.INTER_LANCZOS4
    ) -> np.ndarray:
        """
        Изменение размера изображения
        
        Args:
            image: Исходное изображение
            width: Целевая ширина
            height: Целевая высота
            maintain_aspect: Сохранять пропорции
            interpolation: Метод интерполяции
        
        Returns:
            Изменённое изображение
        """
        try:
            h, w = image.shape[:2]
            
            # Если размеры не заданы, возврат оригинала
            if width is None and height is None:
                return image
            
            # Если нужно сохранить пропорции
            if maintain_aspect:
                if width is not None and height is not None:
                    # Вычисление масштаба для подгонки в заданные размеры
                    scale = min(width / w, height / h)
                    new_w = int(w * scale)
                    new_h = int(h * scale)
                elif width is not None:
                    scale = width / w
                    new_w = width
                    new_h = int(h * scale)
                else:  # height is not None
                    scale = height / h
                    new_h = height
                    new_w = int(w * scale)
            else:
                # Игнорирование пропорций
                new_w = width or w
                new_h = height or h
            
            # Изменение размера
            resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
            
            logger.debug(f"Изображение изменено: {w}x{h} -> {new_w}x{new_h}")
            return resized
        
        except Exception as e:
            logger.error(f"Ошибка при изменении размера изображения: {e}")
            return image
    
    @staticmethod
    def resize_to_max_dimension(
        image: np.ndarray,
        max_dimension: int,
        interpolation: int = cv2.INTER_LANCZOS4
    ) -> np.ndarray:
        """
        Изменение размера с ограничением максимальной стороны
        
        Args:
            image: Исходное изображение
            max_dimension: Максимальный размер стороны
            interpolation: Метод интерполяции
        
        Returns:
            Изменённое изображение
        """
        try:
            h, w = image.shape[:2]
            max_side = max(h, w)
            
            if max_side <= max_dimension:
                return image
            
            scale = max_dimension / max_side
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
            
            logger.info(f"Изображение масштабировано до макс. стороны {max_dimension}px")
            return resized
        
        except Exception as e:
            logger.error(f"Ошибка при масштабировании: {e}")
            return image
    
    @staticmethod
    def resize_to_fit(
        image: np.ndarray,
        target_width: int,
        target_height: int,
        padding_color: Tuple[int, int, int] = (255, 255, 255),
        interpolation: int = cv2.INTER_LANCZOS4
    ) -> np.ndarray:
        """
        Изменение размера с подгонкой в заданные размеры (с padding)
        
        Args:
            image: Исходное изображение
            target_width: Целевая ширина
            target_height: Целевая высота
            padding_color: Цвет padding
            interpolation: Метод интерполяции
        
        Returns:
            Изменённое изображение с padding
        """
        try:
            # Масштабирование с сохранением пропорций
            resized = ImageResizer.resize(
                image,
                target_width,
                target_height,
                maintain_aspect=True,
                interpolation=interpolation
            )
            
            h, w = resized.shape[:2]
            
            # Если размеры уже подходят
            if w == target_width and h == target_height:
                return resized
            
            # Создание холста 
            canvas = np.full((target_height, target_width, 3), padding_color, dtype=np.uint8)
            
            # Вычисление позиции для центрирования
            x_offset = (target_width - w) // 2
            y_offset = (target_height - h) // 2
            
            # Размещение изображения на холсте
            canvas[y_offset:y_offset + h, x_offset:x_offset + w] = resized
            
            logger.info(f"Изображение подогнано к размерам {target_width}x{target_height}")
            return canvas
        
        except Exception as e:
            logger.error(f"Ошибка при подгонке размера: {e}")
            return image
    
    @staticmethod
    def resize_to_fill(
        image: np.ndarray,
        target_width: int,
        target_height: int,
        interpolation: int = cv2.INTER_LANCZOS4
    ) -> np.ndarray:
        """
        Изменение размера с заполнением заданных размеров (с обрезкой)
        
        Args:
            image: Исходное изображение
            target_width: Целевая ширина
            target_height: Целевая высота
            interpolation: Метод интерполяции
        
        Returns:
            Изменённое изображение
        """
        try:
            h, w = image.shape[:2]
            target_ratio = target_width / target_height
            current_ratio = w / h
            
            # Вычисление масштаба для заполнения
            if current_ratio > target_ratio:
                # Масштабируем по высоте
                scale = target_height / h
            else:
                # Масштабируем по ширине
                scale = target_width / w
            
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            # Масштабирование
            resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
            
            # Кадрирование до нужного размера
            if new_w > target_width:
                x_start = (new_w - target_width) // 2
                resized = resized[:, x_start:x_start + target_width]
            
            if new_h > target_height:
                y_start = (new_h - target_height) // 2
                resized = resized[y_start:y_start + target_height, :]
            
            logger.info(f"Изображение заполнено до размеров {target_width}x{target_height}")
            return resized
        
        except Exception as e:
            logger.error(f"Ошибка при заполнении размера: {e}")
            return image
    
    @staticmethod
    def scale_by_factor(
        image: np.ndarray,
        scale_factor: float,
        interpolation: int = cv2.INTER_LANCZOS4
    ) -> np.ndarray:
        """
        Масштабирование изображения на заданный коэффициент
        
        Args:
            image: Исходное изображение
            scale_factor: Коэффициент масштабирования
            interpolation: Метод интерполяции
        
        Returns:
            Масштабированное изображение
        """
        try:
            if scale_factor <= 0:
                logger.error(f"Недопустимый коэффициент масштабирования: {scale_factor}")
                return image
            
            h, w = image.shape[:2]
            new_w = int(w * scale_factor)
            new_h = int(h * scale_factor)
            
            if new_w <= 0 or new_h <= 0:
                logger.error("Размер изображения стал нулевым или отрицательным")
                return image
            
            # Выбор метода интерполяции
            if scale_factor < 1:
                interp = cv2.INTER_AREA  # Лучше для уменьшения
            else:
                interp = interpolation
            
            resized = cv2.resize(image, (new_w, new_h), interpolation=interp)
            
            logger.debug(f"Масштабирование на {scale_factor:.2f}x: {w}x{h} -> {new_w}x{new_h}")
            return resized
        
        except Exception as e:
            logger.error(f"Ошибка при масштабировании: {e}")
            return image
    
    @staticmethod
    def get_optimal_scale(
        image_shape: Tuple[int, int],
        target_width: int,
        target_height: int,
        fit_mode: str = "fit"
    ) -> float:
        """
        Вычисление оптимального масштаба для изображения
        
        Args:
            image_shape: Форма изображения (height, width)
            target_width: Целевая ширина
            target_height: Целевая высота
            fit_mode: Режим подгонки ("fit" или "fill")
        
        Returns:
            Коэффициент масштабирования
        """
        h, w = image_shape
        
        scale_w = target_width / w
        scale_h = target_height / h
        
        if fit_mode == "fit":
            scale = min(scale_w, scale_h)
        else:  # fill
            scale = max(scale_w, scale_h)
        
        return scale