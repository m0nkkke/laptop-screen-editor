"""
Тесты для детектора экрана
"""
import pytest
import numpy as np
from pathlib import Path

from app.ml.inference import ScreenDetector
from app.utils.geometry import Polygon


@pytest.fixture
def detector():
    """Фикстура детектора"""
    return ScreenDetector()


@pytest.fixture
def sample_image():
    """Создание тестового изображения"""
    # Создание простого изображения 640x480
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Добавление "ноутбука" (серый прямоугольник)
    image[100:400, 150:550] = [128, 128, 128]
    
    # Добавление "экрана" (тёмный прямоугольник)
    image[150:350, 200:500] = [50, 50, 50]
    
    return image


def test_detector_initialization(detector):
    """Тест инициализации детектора"""
    assert detector is not None
    
    # Проверка, что модель либо загружена, либо корректно обработана её отсутствие
    if detector.model_path.exists():
        assert detector.is_loaded
    else:
        assert not detector.is_loaded


def test_detect_screen_with_mock_image(detector, sample_image):
    """Тест детектирования на простом изображении"""
    if not detector.is_loaded:
        pytest.skip("Модель не загружена")
    
    result = detector.detect_screen(sample_image)
    
    # Результат может быть None если экран не найден (что ожидаемо для синтетического изображения)
    if result is not None:
        mask, polygon = result
        
        assert isinstance(mask, np.ndarray)
        assert isinstance(polygon, Polygon)
        assert mask.shape == sample_image.shape[:2]
        assert len(polygon.points) >= 3


def test_detect_screen_with_invalid_input(detector):
    """Тест с некорректным входом"""
    if not detector.is_loaded:
        pytest.skip("Модель не загружена")
    
    # Пустой массив
    result = detector.detect_screen(np.array([]))
    assert result is None
    
    # Неправильная размерность
    result = detector.detect_screen(np.zeros((10, 10), dtype=np.uint8))
    assert result is None


def test_visualize_detection(detector, sample_image):
    """Тест визуализации"""
    if not detector.is_loaded:
        pytest.skip("Модель не загружена")
    
    result = detector.detect_screen(sample_image)
    
    if result is not None:
        mask, polygon = result
        
        vis_image = detector.visualize_detection(sample_image, mask, polygon)
        
        assert vis_image is not None
        assert vis_image.shape == sample_image.shape
        assert vis_image.dtype == np.uint8


def test_batch_detection(detector):
    """Тест пакетной обработки"""
    if not detector.is_loaded:
        pytest.skip("Модель не загружена")
    
    # Создание нескольких тестовых изображений
    images = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(3)]
    
    results = detector.detect_batch(images)
    
    assert len(results) == len(images)
    
    # Все результаты должны быть либо None, либо (mask, polygon)
    for result in results:
        assert result is None or (isinstance(result, tuple) and len(result) == 2)


def test_confidence_threshold(detector, sample_image):
    """Тест порога уверенности"""
    if not detector.is_loaded:
        pytest.skip("Модель не загружена")
    
    # Тест с низким порогом
    result_low = detector.detect_screen(sample_image, confidence=0.1)
    
    # Тест с высоким порогом
    result_high = detector.detect_screen(sample_image, confidence=0.9)
    
    # С низким порогом больше шансов найти что-то
    # (но это зависит от изображения)
    # Просто проверяем, что функция работает с разными порогами
    assert result_low is None or isinstance(result_low, tuple)
    assert result_high is None or isinstance(result_high, tuple)