"""
Диалоговые окна приложения
"""
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QGroupBox, QSpinBox, QDoubleSpinBox,
    QCheckBox, QComboBox, QLineEdit, QFormLayout
)
from PySide6.QtCore import Qt
from loguru import logger

from app.config import settings


class SettingsDialog(QDialog):
    """Диалог настроек приложения"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Настройки")
        self.setModal(True)
        self.setMinimumWidth(400)
        
        self.init_ui()
        self.load_settings()
    
    def init_ui(self):
        """Инициализация UI"""
        layout = QVBoxLayout(self)
        
        # Группа: Модель ML
        ml_group = QGroupBox("Настройки модели")
        ml_layout = QFormLayout()
        
        self.spin_confidence = QDoubleSpinBox()
        self.spin_confidence.setRange(0.1, 1.0)
        self.spin_confidence.setSingleStep(0.05)
        self.spin_confidence.setDecimals(2)
        ml_layout.addRow("Порог уверенности:", self.spin_confidence)
        
        self.spin_iou = QDoubleSpinBox()
        self.spin_iou.setRange(0.1, 1.0)
        self.spin_iou.setSingleStep(0.05)
        self.spin_iou.setDecimals(2)
        ml_layout.addRow("IoU порог:", self.spin_iou)
        
        ml_group.setLayout(ml_layout)
        layout.addWidget(ml_group)
        
        # Группа: Выходные файлы
        output_group = QGroupBox("Настройки выходных файлов")
        output_layout = QFormLayout()
        
        self.spin_jpeg_quality = QSpinBox()
        self.spin_jpeg_quality.setRange(1, 100)
        output_layout.addRow("Качество JPEG:", self.spin_jpeg_quality)
        
        self.spin_png_compression = QSpinBox()
        self.spin_png_compression.setRange(0, 9)
        output_layout.addRow("Сжатие PNG:", self.spin_png_compression)
        
        self.edit_output_suffix = QLineEdit()
        output_layout.addRow("Суффикс файлов:", self.edit_output_suffix)
        
        output_group.setLayout(output_layout)
        layout.addWidget(output_group)
        
        # Группа: Производительность
        perf_group = QGroupBox("Производительность")
        perf_layout = QFormLayout()
        
        self.spin_max_threads = QSpinBox()
        self.spin_max_threads.setRange(1, 16)
        perf_layout.addRow("Макс. потоков:", self.spin_max_threads)
        
        self.spin_batch_size = QSpinBox()
        self.spin_batch_size.setRange(1, 100)
        perf_layout.addRow("Размер пакета:", self.spin_batch_size)
        
        perf_group.setLayout(perf_layout)
        layout.addWidget(perf_group)
        
        # Кнопки
        btn_layout = QHBoxLayout()
        
        self.btn_ok = QPushButton("OK")
        self.btn_cancel = QPushButton("Отмена")
        self.btn_reset = QPushButton("По умолчанию")
        
        self.btn_ok.clicked.connect(self.accept)
        self.btn_cancel.clicked.connect(self.reject)
        self.btn_reset.clicked.connect(self.reset_to_defaults)
        
        btn_layout.addWidget(self.btn_reset)
        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_ok)
        btn_layout.addWidget(self.btn_cancel)
        
        layout.addLayout(btn_layout)
    
    def load_settings(self):
        """Загрузка текущих настроек"""
        self.spin_confidence.setValue(settings.MODEL_CONFIDENCE)
        self.spin_iou.setValue(settings.MODEL_IOU_THRESHOLD)
        self.spin_jpeg_quality.setValue(settings.JPEG_QUALITY)
        self.spin_png_compression.setValue(settings.PNG_COMPRESSION)
        self.edit_output_suffix.setText("_processed")
        self.spin_max_threads.setValue(settings.MAX_THREADS)
        self.spin_batch_size.setValue(settings.BATCH_SIZE)
    
    def reset_to_defaults(self):
        """Сброс настроек на значения по умолчанию"""
        self.spin_confidence.setValue(0.25)
        self.spin_iou.setValue(0.45)
        self.spin_jpeg_quality.setValue(95)
        self.spin_png_compression.setValue(6)
        self.edit_output_suffix.setText("_processed")
        self.spin_max_threads.setValue(4)
        self.spin_batch_size.setValue(10)
    
    def get_settings(self) -> dict:
        """Получение настроек из диалога"""
        return {
            'model_confidence': self.spin_confidence.value(),
            'model_iou': self.spin_iou.value(),
            'jpeg_quality': self.spin_jpeg_quality.value(),
            'png_compression': self.spin_png_compression.value(),
            'output_suffix': self.edit_output_suffix.text(),
            'max_threads': self.spin_max_threads.value(),
            'batch_size': self.spin_batch_size.value()
        }


class AboutDialog(QDialog):
    """Диалог "О программе" """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("О программе")
        self.setModal(True)
        self.setMinimumWidth(400)
        
        self.init_ui()
    
    def init_ui(self):
        """Инициализация UI"""
        layout = QVBoxLayout(self)
        
        # Название
        title = QLabel("<h2>Laptop Screen Editor</h2>")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Версия
        version = QLabel("Версия 0.1.0")
        version.setAlignment(Qt.AlignCenter)
        layout.addWidget(version)
        
        # Описание
        description = QLabel(
            "Автоматизированная обработка изображений ноутбуков "
            "для карточек товаров.<br><br>"
            "Возможности:<br>"
            "• Автоматическое определение экрана<br>"
            "• Замена/заливка содержимого экрана<br>"
            "• Пакетная обработка изображений<br>"
            "• Кадрирование и масштабирование"
        )
        description.setWordWrap(True)
        description.setAlignment(Qt.AlignCenter)
        layout.addWidget(description)
        
        layout.addSpacing(20)
        
        # Технологии
        tech = QLabel(
            "<b>Технологии:</b><br>"
            "Python 3.11+ • PySide6 • OpenCV<br>"
            "YOLOv8 • Pillow • NumPy"
        )
        tech.setAlignment(Qt.AlignCenter)
        layout.addWidget(tech)
        
        layout.addSpacing(20)
        
        # Кнопка закрытия
        btn_close = QPushButton("Закрыть")
        btn_close.clicked.connect(self.accept)
        layout.addWidget(btn_close)


class ProcessingProgressDialog(QDialog):
    """Диалог прогресса обработки"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Обработка изображений")
        self.setModal(True)
        self.setMinimumWidth(400)
        
        self.init_ui()
    
    def init_ui(self):
        """Инициализация UI"""
        from PySide6.QtWidgets import QProgressBar, QTextEdit
        
        layout = QVBoxLayout(self)
        
        # Основная информация
        self.label_status = QLabel("Подготовка к обработке...")
        layout.addWidget(self.label_status)
        
        # Прогресс бар
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)
        
        # Лог обработки
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(200)
        layout.addWidget(self.log_text)
        
        # Кнопка отмены
        self.btn_cancel = QPushButton("Отмена")
        self.btn_cancel.clicked.connect(self.reject)
        layout.addWidget(self.btn_cancel)
    
    def update_progress(self, current: int, total: int, message: str = ""):
        """Обновление прогресса"""
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)
        
        if message:
            self.label_status.setText(f"Обработка {current}/{total}")
            self.log_text.append(message)
    
    def set_completed(self, success: bool, message: str = ""):
        """Установка статуса завершения"""
        if success:
            self.label_status.setText("Обработка завершена успешно")
        else:
            self.label_status.setText("Обработка завершена с ошибками")
        
        if message:
            self.log_text.append(f"\n{message}")
        
        self.btn_cancel.setText("Закрыть")