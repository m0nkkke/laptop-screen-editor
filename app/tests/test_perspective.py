"""
Тесты для перспективных преобразований
"""
import pytest
import numpy as np

from app.core.perspective import PerspectiveTransformer
from app.utils.geometry import Polygon, order_points, calculate_perspective_size


@pytest.fixture
def sample_image():
    """Создание тестового изображения"""
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_polygon():
    """Создание тестового полигона (четырёхугольник)"""
    points = [
        (100.0, 100.0),  # top-left
        (500.0, 100.0),  # top-right
        (500.0, 400.0),  # bottom-right
        (100.0, 400.0)   # bottom-left
    ]
    return Polygon(points)


@pytest.fixture
def trapezoid_polygon():
    """Создание трапециевидного полигона (с перспективой)"""
    points = [
        (150.0, 100.0),  # top-left
        (450.0, 100.0),  # top-right
        (500.0, 400.0),  # bottom-right
        (100.0, 400.0)   # bottom-left
    ]
    return Polygon(points)


def test_order_points():
    """Тест упорядочивания точек"""
    # Неупорядоченные точки
    points = np.array([
        [500.0, 400.0],  # bottom-right
        [100.0, 100.0],  # top-left
        [500.0, 100.0],  # top-right
        [100.0, 400.0]   # bottom-left
    ])
    
    ordered = order_points(points)
    
    # Проверка порядка: tl, tr, br, bl
    assert ordered[0][0] < ordered[1][0]  # tl.x < tr.x
    assert ordered[0][1] < ordered[2][1]  # tl.y < br.y


def test_calculate_perspective_size(sample_polygon):
    """Тест вычисления размера при перспективе"""
    points = sample_polygon.to_numpy()
    width, height = calculate_perspective_size(points)
    
    assert width > 0
    assert height > 0
    assert isinstance(width, int)
    assert isinstance(height, int)


def test_extract_screen_region(sample_image, sample_polygon):
    """Тест извлечения области экрана"""
    transformer = PerspectiveTransformer()
    
    result = transformer.extract_screen_region(sample_image, sample_polygon)
    
    assert result is not None
    assert isinstance(result, np.ndarray)
    assert len(result.shape) == 3  # RGB изображение
    assert result.shape[2] == 3


def test_extract_screen_region_with_perspective(sample_image, trapezoid_polygon):
    """Тест извлечения области с перспективой"""
    transformer = PerspectiveTransformer()
    
    result = transformer.extract_screen_region(sample_image, trapezoid_polygon)
    
    assert result is not None
    assert isinstance(result, np.ndarray)


def test_extract_screen_region_invalid_polygon(sample_image):
    """Тест с невалидным полигоном"""
    transformer = PerspectiveTransformer()
    
    # Полигон с недостаточным количеством точек
    invalid_polygon = Polygon([(100.0, 100.0), (200.0, 200.0)])
    
    result = transformer.extract_screen_region(sample_image, invalid_polygon)
    
    assert result is None


def test_warp_image_to_polygon(sample_image, sample_polygon):
    """Тест применения перспективы к изображению"""
    transformer = PerspectiveTransformer()
    
    # Создание тестового изображения для трансформации
    test_image = np.ones((300, 400, 3), dtype=np.uint8) * 128
    
    result = transformer.warp_image_to_polygon(
        test_image,
        sample_polygon,
        (sample_image.shape[0], sample_image.shape[1])
    )
    
    assert result is not None
    assert result.shape == sample_image.shape


def test_create_perspective_mask(sample_polygon):
    """Тест создания маски"""
    transformer = PerspectiveTransformer()
    
    image_shape = (480, 640)
    mask = transformer.create_perspective_mask(sample_polygon, image_shape)
    
    assert mask is not None
    assert mask.shape == image_shape
    assert mask.dtype == np.uint8
    assert np.any(mask > 0)  # Маска не должна быть пустой


def test_correct_perspective_distortion(sample_image, trapezoid_polygon):
    """Тест исправления перспективы"""
    transformer = PerspectiveTransformer()
    
    result = transformer.correct_perspective_distortion(
        sample_image,
        trapezoid_polygon
    )
    
    assert result is not None
    assert isinstance(result, np.ndarray)


def test_correct_perspective_with_aspect_ratio(sample_image, trapezoid_polygon):
    """Тест исправления перспективы с заданным соотношением сторон"""
    transformer = PerspectiveTransformer()
    
    result = transformer.correct_perspective_distortion(
        sample_image,
        trapezoid_polygon,
        target_aspect_ratio=16/9
    )
    
    assert result is not None
    
    # Проверка соотношения сторон (с небольшой погрешностью)
    h, w = result.shape[:2]
    actual_ratio = w / h
    expected_ratio = 16 / 9
    assert abs(actual_ratio - expected_ratio) < 0.1


def test_apply_homography():
    """Тест применения гомографии"""
    transformer = PerspectiveTransformer()
    
    # Тестовое изображение
    image = np.ones((300, 400, 3), dtype=np.uint8) * 128
    
    # Исходные и целевые точки
    src_points = np.array([
        [0, 0],
        [399, 0],
        [399, 299],
        [0, 299]
    ], dtype=np.float32)
    
    dst_points = np.array([
        [50, 50],
        [350, 50],
        [350, 250],
        [50, 250]
    ], dtype=np.float32)
    
    result = transformer.apply_homography(
        image,
        src_points,
        dst_points,
        (480, 640)
    )
    
    assert result is not None
    assert result.shape == (480, 640, 3)


def test_apply_homography_invalid_points():
    """Тест гомографии с невалидными точками"""
    transformer = PerspectiveTransformer()
    
    image = np.ones((300, 400, 3), dtype=np.uint8)
    
    # Недостаточно точек
    src_points = np.array([[0, 0], [100, 100]])
    dst_points = np.array([[50, 50], [150, 150]])
    
    result = transformer.apply_homography(
        image,
        src_points,
        dst_points,
        (480, 640)
    )
    
    assert result is None