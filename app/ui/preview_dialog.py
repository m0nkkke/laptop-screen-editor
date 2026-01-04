"""
Окно предпросмотра для проверки и корректировки обнаруженных областей
"""
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QWidget, QMessageBox, QCheckBox
)
from PySide6.QtCore import Qt, Signal
from loguru import logger

from app.core.image_loader import ImageLoader
from app.ml.inference import ScreenDetector
from app.ui.image_viewer import InteractiveImageViewer
from app.utils.geometry import Polygon
from app.core.screen_editor import ScreenEditor


class DetectionPreviewDialog(QDialog):
    """
    Диалог предпросмотра обнаруженных областей с возможностью корректировки
    """
    
    # Сигнал для передачи результатов (file_path, polygon, mask, skip)
    result_ready = Signal(Path, object, object, bool)
    
    def __init__(self, files: List[Path], detector: ScreenDetector, parent=None):
        super().__init__(parent)
        
        self.files = files
        self.detector = detector
        self.current_index = 0
        self.results = {}  # {file_path: (polygon, mask, skip)}
        
        self.current_image: Optional[np.ndarray] = None
        self.current_mask: Optional[np.ndarray] = None
        self.current_polygon: Optional[Polygon] = None
        self.original_polygon: Optional[Polygon] = None
        
        self.init_ui()
        self.load_current_image()
    
    def init_ui(self):
        """Инициализация UI"""
        self.setWindowTitle("Проверка обнаружения экранов")
        self.setModal(True)  # Модальное окно
        
        if parent := self.parent():
            parent_geometry = parent.geometry()
            self.setGeometry(parent_geometry)
            self.showMaximized()
        else:
            self.showMaximized()
        
        layout = QVBoxLayout(self)
        
        # Информация о текущем файле
        info_layout = QHBoxLayout()
        
        self.label_progress = QLabel()
        info_layout.addWidget(self.label_progress)
        
        info_layout.addStretch()
        
        self.label_filename = QLabel()
        self.label_filename.setStyleSheet("font-weight: bold;")
        info_layout.addWidget(self.label_filename)
        
        layout.addLayout(info_layout)
        
        # Вьюер изображений
        self.viewer = InteractiveImageViewer()
        layout.addWidget(self.viewer)
        
        # Панель управления zoom
        zoom_layout = QHBoxLayout()
        zoom_layout.addWidget(QLabel("🔍 Масштаб:"))
        
        self.btn_zoom_in = QPushButton("➕")
        self.btn_zoom_in.setToolTip("Увеличить (Ctrl + колесо мыши вверх)")
        self.btn_zoom_in.setFixedWidth(40)
        self.btn_zoom_in.clicked.connect(self.viewer.zoom_in)
        zoom_layout.addWidget(self.btn_zoom_in)
        
        self.btn_zoom_out = QPushButton("➖")
        self.btn_zoom_out.setToolTip("Уменьшить (Ctrl + колесо мыши вниз)")
        self.btn_zoom_out.setFixedWidth(40)
        self.btn_zoom_out.clicked.connect(self.viewer.zoom_out)
        zoom_layout.addWidget(self.btn_zoom_out)
        
        self.btn_zoom_reset = QPushButton("⤢ Сбросить масштаб")
        self.btn_zoom_reset.setToolTip("Подогнать под размер окна")
        self.btn_zoom_reset.clicked.connect(self.viewer.reset_zoom)
        zoom_layout.addWidget(self.btn_zoom_reset)
        
        zoom_layout.addStretch()
        layout.addLayout(zoom_layout)
        
        # Статус обнаружения
        self.label_detection = QLabel()
        layout.addWidget(self.label_detection)
        
        # Чекбокс автоматического перехода
        self.cb_auto_next = QCheckBox("Автоматически переходить к следующему")
        self.cb_auto_next.setChecked(True)
        layout.addWidget(self.cb_auto_next)
        
        # Кнопки управления
        btn_layout = QHBoxLayout()
        
        # Кнопка "Пропустить"
        self.btn_skip = QPushButton("⏭ Пропустить файл")
        self.btn_skip.setToolTip("Не обрабатывать этот файл")
        self.btn_skip.clicked.connect(self.skip_file)
        btn_layout.addWidget(self.btn_skip)
        
        btn_layout.addStretch()
        
        # Кнопка "Сбросить"
        self.btn_reset = QPushButton("🔄 Сбросить изменения")
        self.btn_reset.setToolTip("Вернуть исходное обнаружение")
        self.btn_reset.clicked.connect(self.reset_polygon)
        self.btn_reset.setEnabled(False)
        btn_layout.addWidget(self.btn_reset)
        
        # Кнопка "Назад"
        self.btn_prev = QPushButton("◀ Назад")
        self.btn_prev.clicked.connect(self.previous_image)
        self.btn_prev.setEnabled(False)
        btn_layout.addWidget(self.btn_prev)
        
        # Кнопка "Принять и продолжить"
        self.btn_next = QPushButton("Принять и продолжить ▶")
        self.btn_next.setDefault(True)
        self.btn_next.clicked.connect(self.accept_and_next)
        self.btn_next.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        btn_layout.addWidget(self.btn_next)
        
        # Кнопка "Завершить"
        self.btn_finish = QPushButton("✓ Завершить проверку")
        self.btn_finish.clicked.connect(self.finish_preview)
        self.btn_finish.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #0b7dda;
            }
        """)
        btn_layout.addWidget(self.btn_finish)
        
        layout.addLayout(btn_layout)
        
        # Включение режима редактирования
        self.viewer.enable_editing(True)
    
    def load_current_image(self):
        """Загрузка текущего изображения"""
        if self.current_index >= len(self.files):
            self.finish_preview()
            return
        
        file_path = self.files[self.current_index]
        
        try:
            # Обновление UI
            self.label_progress.setText(
                f"Файл {self.current_index + 1} из {len(self.files)}"
            )
            self.label_filename.setText(file_path.name)
            
            # Загрузка изображения
            logger.info(f"Загрузка для проверки: {file_path.name}")
            
            # Очистка viewer от предыдущего изображения
            self.viewer.clear()
            
            self.current_image = ImageLoader.load_image(file_path)
            
            if self.current_image is None:
                QMessageBox.warning(
                    self,
                    "Ошибка",
                    f"Не удалось загрузить изображение:\n{file_path.name}"
                )
                self.skip_file()
                return
            
            # Изменение размера для превью
            preview_image = ImageLoader.resize_for_preview(self.current_image)
            scale_factor = preview_image.shape[1] / self.current_image.shape[1]
            
            # Детектирование экрана
            detection = self.detector.detect_screen(self.current_image)
            
            # Наличие сохранённых изменений для файла
            if file_path in self.results:
                saved_polygon, saved_mask, skip = self.results[file_path]
                
                if skip:
                    # Файл был пропущен ранее
                    self.label_detection.setText("⏭ Файл был пропущен ранее")
                    self.label_detection.setStyleSheet("color: #ff9800; font-weight: bold;")
                    
                    self.viewer.display_image(preview_image)
                    self.viewer.set_scale_factor(scale_factor)
                    
                    self.current_mask = None
                    self.current_polygon = None
                    self.original_polygon = None
                    self.btn_reset.setEnabled(False)
                elif saved_polygon is not None:
                    # Есть сохранённые изменения
                    self.label_detection.setText("✏️ Используются ваши изменения")
                    self.label_detection.setStyleSheet("color: #2196F3; font-weight: bold;")
                    
                    self.current_mask = saved_mask
                    self.current_polygon = saved_polygon
                    self.original_polygon = Polygon([p.to_tuple() for p in saved_polygon.points])
                    
                    # Масштабирование для превью
                    preview_polygon = saved_polygon.scale(scale_factor, scale_factor)
                    
                    self.viewer.display_image(preview_image)
                    self.viewer.set_scale_factor(scale_factor)
                    self.viewer.display_polygon(preview_polygon)
                    
                    self.btn_reset.setEnabled(True)
                else:
                    # Обычная логика детектирования
                    self._handle_detection(detection, preview_image, scale_factor)
            elif detection is None:
                # Экран не обнаружен
                self._handle_detection(None, preview_image, scale_factor)
            else:
                # Обычная логика детектирования
                self._handle_detection(detection, preview_image, scale_factor)
            
            # Обновление кнопок навигации
            self.btn_prev.setEnabled(self.current_index > 0)
            
        except Exception as e:
            logger.error(f"Ошибка загрузки изображения: {e}")
            QMessageBox.critical(
                self,
                "Ошибка",
                f"Ошибка при загрузке изображения:\n{str(e)}"
            )
            self.skip_file()
    
    def reset_polygon(self):
        """Сброс полигона к исходному"""
        if self.original_polygon is None:
            return
        
        # Восстановление оригинального полигона
        self.current_polygon = Polygon([p.to_tuple() for p in self.original_polygon.points])
        
        # Масштабирование для превью
        scale_factor = self.viewer.scale_factor
        preview_polygon = self.current_polygon.scale(scale_factor, scale_factor)
        
        # Перерисовка
        self.viewer.display_polygon(preview_polygon)
        
        logger.info("Полигон сброшен к исходному")
    
    def accept_and_next(self):
        """Принять текущую область и перейти к следующему"""
        file_path = self.files[self.current_index]
        
        # Получение отредактированного полигона
        edited_polygon = self.viewer.get_edited_polygon()
        
        # Если полигон был отредактирован, пересоздаём маску
        if edited_polygon is not None:
            mask = ScreenEditor.create_screen_mask_from_polygon(
                (self.current_image.shape[0], self.current_image.shape[1]),
                edited_polygon
            )
            polygon = edited_polygon
        else:
            mask = self.current_mask
            polygon = self.current_polygon
        
        # Сохранение результата
        self.results[file_path] = (polygon, mask, False)
        self.result_ready.emit(file_path, polygon, mask, False)
        
        logger.info(f"Принято: {file_path.name}")
        
        # Переход к следующему
        if self.cb_auto_next.isChecked():
            self.next_image()
        else:
            self.label_detection.setText("✓ Принято")
            self.label_detection.setStyleSheet("color: #4CAF50; font-weight: bold;")
    
    def skip_file(self):
        """Пропустить текущий файл"""
        file_path = self.files[self.current_index]
        
        # Сохранение как пропущенный
        self.results[file_path] = (None, None, True)
        self.result_ready.emit(file_path, None, None, True)
        
        logger.info(f"Пропущено: {file_path.name}")
        
        # Переход к следующему
        self.next_image()
    
    def next_image(self):
        """Переход к следующему изображению"""
        # Сохраняем текущие изменения перед переходом
        self._save_current_edits()
        
        self.current_index += 1
        
        if self.current_index >= len(self.files):
            self.finish_preview()
        else:
            self.load_current_image()
    
    def previous_image(self):
        """Переход к предыдущему изображению"""
        # Сохраняем текущие изменения перед переходом
        self._save_current_edits()
        
        if self.current_index > 0:
            self.current_index -= 1
            self.load_current_image()
    
    def _save_current_edits(self):
        """Сохранение текущих изменений"""
        if self.current_index < len(self.files):
            file_path = self.files[self.current_index]
            
            if file_path in self.results and self.results[file_path][2] is True:
                return
            # Отредактированный полигон
            edited_polygon = self.viewer.get_edited_polygon()
            
            # Если полигон был отредактирован - пересоздать маску
            if edited_polygon is not None:
                mask = ScreenEditor.create_screen_mask_from_polygon(
                    (self.current_image.shape[0], self.current_image.shape[1]),
                    edited_polygon
                )
                polygon = edited_polygon
            else:
                mask = self.current_mask
                polygon = self.current_polygon
            
            # Сохранение в results только (есть данные)
            if polygon is not None and mask is not None:
                self.results[file_path] = (polygon, mask, False)
    
    def _handle_detection(self, detection, preview_image, scale_factor):
        """Обработка результата детектирования"""
        if detection is None:
            self.label_detection.setText("⚠️ Экран не обнаружен автоматически")
            self.label_detection.setStyleSheet("color: #ff9800; font-weight: bold;")
            
            # Отображение без полигона
            self.viewer.display_image(preview_image)
            self.viewer.set_scale_factor(scale_factor)
            
            self.current_mask = None
            self.current_polygon = None
            self.original_polygon = None
            
            self.btn_reset.setEnabled(False)
        else:
            mask, polygon = detection
            
            self.label_detection.setText("✓ Экран обнаружен автоматически")
            self.label_detection.setStyleSheet("color: #4CAF50; font-weight: bold;")
            
            # Сохранение оригиналов
            self.current_mask = mask
            self.current_polygon = polygon
            self.original_polygon = Polygon([p.to_tuple() for p in polygon.points])
            
            # Масштабирование полигона для превью
            preview_polygon = polygon.scale(scale_factor, scale_factor)
            
            # Отображение
            self.viewer.display_image(preview_image)
            self.viewer.set_scale_factor(scale_factor)
            self.viewer.display_polygon(preview_polygon)
            
            self.btn_reset.setEnabled(True)
    
    def finish_preview(self):
        """Завершение проверки"""
        # Подсчёт статистики
        total = len(self.files)
        accepted = sum(1 for _, _, skip in self.results.values() if not skip)
        skipped = sum(1 for _, _, skip in self.results.values() if skip)
        remaining = total - accepted - skipped
        
        if remaining > 0:
            reply = QMessageBox.question(
                self,
                "Завершить проверку?",
                f"Осталось непроверенных файлов: {remaining}\n\n"
                f"Принято: {accepted}\n"
                f"Пропущено: {skipped}\n\n"
                f"Завершить проверку?",
                QMessageBox.Yes | QMessageBox.No
            )
            
            if reply == QMessageBox.No:
                return
        
        logger.info(f"Проверка завершена: {accepted} принято, {skipped} пропущено")
        self.accept()
    
    def get_results(self):
        """Получение результатов проверки"""
        return self.results
    
    def closeEvent(self, event):
        """Обработка закрытия окна"""
        accepted = sum(1 for _, _, skip in self.results.values() if not skip)
        remaining = len(self.files) - len(self.results)
        
        if remaining > 0:
            reply = QMessageBox.question(
                self,
                "Закрыть окно?",
                f"Проверено: {accepted} файлов\n"
                f"Осталось: {remaining} файлов\n\n"
                f"Закрыть окно проверки?",
                QMessageBox.Yes | QMessageBox.No
            )
            
            if reply == QMessageBox.No:
                event.ignore()
                return
        
        event.accept()