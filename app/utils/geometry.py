"""
Утилиты для работы с геометрией: точки, полигоны, bounding boxes
"""
import numpy as np
from typing import List, Tuple, Optional


class Point:
    """Класс для представления точки"""
    
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y
    
    def to_tuple(self) -> Tuple[float, float]:
        return (self.x, self.y)
    
    def to_int_tuple(self) -> Tuple[int, int]:
        return (int(self.x), int(self.y))
    
    def __repr__(self):
        return f"Point({self.x}, {self.y})"


class Polygon:
    """Класс для работы с полигонами"""
    
    def __init__(self, points: List[Tuple[float, float]]):
        self.points = [Point(x, y) for x, y in points]
    
    def to_numpy(self) -> np.ndarray:
        """Конвертация в numpy array"""
        return np.array([p.to_tuple() for p in self.points], dtype=np.float32)
    
    def to_int_numpy(self) -> np.ndarray:
        """Конвертация в numpy array с целыми числами"""
        return np.array([p.to_int_tuple() for p in self.points], dtype=np.int32)
    
    def get_bounding_box(self) -> Tuple[int, int, int, int]:
        """Получение bounding box (x, y, w, h)"""
        points_np = self.to_numpy()
        x_min = int(np.min(points_np[:, 0]))
        y_min = int(np.min(points_np[:, 1]))
        x_max = int(np.max(points_np[:, 0]))
        y_max = int(np.max(points_np[:, 1]))
        return x_min, y_min, x_max - x_min, y_max - y_min
    
    def scale(self, scale_x: float, scale_y: float) -> 'Polygon':
        """Масштабирование полигона"""
        scaled_points = [(p.x * scale_x, p.y * scale_y) for p in self.points]
        return Polygon(scaled_points)
    
    def __len__(self):
        return len(self.points)
    
    def __repr__(self):
        return f"Polygon({len(self.points)} points)"


class BoundingBox:
    """Класс для работы с bounding box"""
    
    def __init__(self, x: int, y: int, width: int, height: int):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
    
    @property
    def x1(self) -> int:
        return self.x
    
    @property
    def y1(self) -> int:
        return self.y
    
    @property
    def x2(self) -> int:
        return self.x + self.width
    
    @property
    def y2(self) -> int:
        return self.y + self.height
    
    @property
    def center(self) -> Tuple[int, int]:
        return (self.x + self.width // 2, self.y + self.height // 2)
    
    @property
    def area(self) -> int:
        return self.width * self.height
    
    def to_tuple(self) -> Tuple[int, int, int, int]:
        """Возвращает (x, y, w, h)"""
        return (self.x, self.y, self.width, self.height)
    
    def to_xyxy(self) -> Tuple[int, int, int, int]:
        """Возвращает (x1, y1, x2, y2)"""
        return (self.x, self.y, self.x2, self.y2)
    
    def scale(self, scale_x: float, scale_y: float) -> 'BoundingBox':
        """Масштабирование bbox"""
        return BoundingBox(
            int(self.x * scale_x),
            int(self.y * scale_y),
            int(self.width * scale_x),
            int(self.height * scale_y)
        )
    
    def expand(self, margin: int) -> 'BoundingBox':
        """Расширение bbox на заданный margin"""
        return BoundingBox(
            max(0, self.x - margin),
            max(0, self.y - margin),
            self.width + 2 * margin,
            self.height + 2 * margin
        )
    
    def intersection(self, other: 'BoundingBox') -> Optional['BoundingBox']:
        """Пересечение двух bbox"""
        x1 = max(self.x1, other.x1)
        y1 = max(self.y1, other.y1)
        x2 = min(self.x2, other.x2)
        y2 = min(self.y2, other.y2)
        
        if x2 <= x1 or y2 <= y1:
            return None
        
        return BoundingBox(x1, y1, x2 - x1, y2 - y1)
    
    def iou(self, other: 'BoundingBox') -> float:
        """Intersection over Union"""
        intersection = self.intersection(other)
        if intersection is None:
            return 0.0
        
        intersection_area = intersection.area
        union_area = self.area + other.area - intersection_area
        
        return intersection_area / union_area if union_area > 0 else 0.0
    
    def __repr__(self):
        return f"BoundingBox(x={self.x}, y={self.y}, w={self.width}, h={self.height})"


def order_points(points: np.ndarray) -> np.ndarray:
    """
    Упорядочивание точек четырёхугольника:
    [top-left, top-right, bottom-right, bottom-left]
    """
    # Сортировка по сумме координат (top-left имеет наименьшую сумму)
    rect = np.zeros((4, 2), dtype=np.float32)
    
    s = points.sum(axis=1)
    rect[0] = points[np.argmin(s)]  # top-left
    rect[2] = points[np.argmax(s)]  # bottom-right
    
    # Разность координат
    diff = np.diff(points, axis=1)
    rect[1] = points[np.argmin(diff)]  # top-right
    rect[3] = points[np.argmax(diff)]  # bottom-left
    
    return rect


def calculate_perspective_size(points: np.ndarray) -> Tuple[int, int]:
    """
    Вычисление размера выходного изображения для перспективной трансформации
    """
    rect = order_points(points)
    
    (tl, tr, br, bl) = rect
    
    # Вычисление ширины
    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = int(max(width_a, width_b))
    
    # Вычисление высоты
    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = int(max(height_a, height_b))
    
    return max_width, max_height


def polygon_to_mask(polygon: Polygon, image_shape: Tuple[int, int]) -> np.ndarray:
    """
    Создание маски из полигона
    
    Args:
        polygon: Полигон
        image_shape: Форма изображения (height, width)
    
    Returns:
        Бинарная маска
    """
    import cv2
    
    mask = np.zeros(image_shape, dtype=np.uint8)
    points = polygon.to_int_numpy().reshape((-1, 1, 2))
    cv2.fillPoly(mask, [points], 255)
    
    return mask


def mask_to_polygon(mask: np.ndarray, epsilon_factor: float = 0.01) -> Optional[Polygon]:
    """
    Извлечение полигона из маски
    
    Args:
        mask: Бинарная маска
        epsilon_factor: Фактор для аппроксимации полигона
    
    Returns:
        Полигон или None
    """
    import cv2
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Берём самый большой контур
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Аппроксимация полигона
    epsilon = epsilon_factor * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
    
    points = [(float(p[0][0]), float(p[0][1])) for p in approx]
    
    return Polygon(points)