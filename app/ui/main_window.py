"""
Главное окно приложения
"""
from pathlib import Path
from typing import List, Optional
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QProgressBar,
    QGroupBox, QRadioButton, QSpinBox, QDoubleSpinBox,
    QCheckBox, QComboBox, QMessageBox, QSplitter,
    QListWidget, QListWidgetItem, QDialog
)
from PySide6.QtCore import Qt, Signal, QThread
from PySide6.QtGui import QIcon, QPixmap
from loguru import logger

from app.config import settings
from app.ml.inference import ScreenDetector
from app.pipeline.processor import ImageProcessor, ProcessingOptions, ScreenFillMode
from app.ui.image_viewer import ImageViewer
from app.ui.dialogs import SettingsDialog
from app.utils.file_utils import get_supported_files, validate_output_directory


class ProcessingThread(QThread):
    """Поток для обработки изображений"""
    progress = Signal(int, int, str)  # current, total, filename
    finished = Signal(list)  # results
    
    def __init__(self, processor, input_files, output_dir, options):
        super().__init__()
        self.processor = processor
        self.input_files = input_files
        self.output_dir = output_dir
        self.options = options
    
    def run(self):
        results = self.processor.process_batch(
            self.input_files,
            self.output_dir,
            self.options,
            progress_callback=self.progress.emit
        )
        self.finished.emit(results)


class MainWindow(QMainWindow):
    """Главное окно приложения"""
    
    def __init__(self):
        super().__init__()
        self.detector = None
        self.processor = None
        self.input_files: List[Path] = []
        self.output_dir: Optional[Path] = None
        self.fill_image_path: Optional[Path] = None
        
        self.processing_thread: Optional[ProcessingThread] = None
        
        self.init_ui()
        self.init_ml()
        
        logger.info("Главное окно инициализировано")
    
    def init_ui(self):
        """Инициализация UI"""
        self.setWindowTitle(settings.WINDOW_TITLE)
        self.setMinimumSize(settings.WINDOW_MIN_WIDTH, settings.WINDOW_MIN_HEIGHT)
        
        # Центральный виджет
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Главный layout
        main_layout = QHBoxLayout(central_widget)
        
        # Левая панель (списки файлов и настройки)
        left_panel = self.create_left_panel()
        
        # Правая панель (просмотр изображений)
        right_panel = self.create_right_panel()
        
        # Splitter для изменения размеров панелей
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([400, 600])
        
        main_layout.addWidget(splitter)
        
        # Строка состояния
        self.statusBar().showMessage("Готов к работе")
    
    def create_left_panel(self) -> QWidget:
        """Создание левой панели"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Группа: Входные файлы
        input_group = QGroupBox("Входные файлы")
        input_layout = QVBoxLayout()
        
        # Кнопки управления файлами
        btn_layout = QHBoxLayout()
        self.btn_add_files = QPushButton("Добавить файлы")
        self.btn_add_folder = QPushButton("Добавить папку")
        self.btn_clear = QPushButton("Очистить")
        
        self.btn_add_files.clicked.connect(self.add_files)
        self.btn_add_folder.clicked.connect(self.add_folder)
        self.btn_clear.clicked.connect(self.clear_files)
        
        btn_layout.addWidget(self.btn_add_files)
        btn_layout.addWidget(self.btn_add_folder)
        btn_layout.addWidget(self.btn_clear)
        
        input_layout.addLayout(btn_layout)
        
        # Список файлов
        self.file_list = QListWidget()
        self.file_list.itemClicked.connect(self.on_file_selected)
        input_layout.addWidget(self.file_list)
        
        input_group.setLayout(input_layout)
        layout.addWidget(input_group)
        
        # Группа: Настройки обработки
        settings_group = QGroupBox("Настройки обработки")
        settings_layout = QVBoxLayout()
        
        # Режим заливки экрана
        fill_layout = QVBoxLayout()
        fill_layout.addWidget(QLabel("Режим обработки экрана:"))
        
        self.rb_black = QRadioButton("Заливка чёрным")
        self.rb_image = QRadioButton("Замена изображением")
        self.rb_none = QRadioButton("Без изменений")
        self.rb_black.setChecked(True)
        
        fill_layout.addWidget(self.rb_black)
        fill_layout.addWidget(self.rb_image)
        fill_layout.addWidget(self.rb_none)
        
        # Кнопка выбора изображения для заливки
        self.btn_select_fill = QPushButton("Выбрать изображение")
        self.btn_select_fill.clicked.connect(self.select_fill_image)
        self.btn_select_fill.setEnabled(False)
        self.rb_image.toggled.connect(lambda checked: self.btn_select_fill.setEnabled(checked))
        fill_layout.addWidget(self.btn_select_fill)
        
        settings_layout.addLayout(fill_layout)
        
        # Перспектива
        self.cb_perspective = QCheckBox("Использовать перспективу")
        self.cb_perspective.setChecked(True)
        settings_layout.addWidget(self.cb_perspective)
        
        # Автокадрирование
        self.cb_auto_crop = QCheckBox("Автоматическое кадрирование")
        settings_layout.addWidget(self.cb_auto_crop)
        
        # Изменение размера
        self.cb_resize = QCheckBox("Изменить размер")
        settings_layout.addWidget(self.cb_resize)
        
        # Размеры
        size_layout = QHBoxLayout()
        size_layout.addWidget(QLabel("Ширина:"))
        self.spin_width = QSpinBox()
        self.spin_width.setRange(100, 10000)
        self.spin_width.setValue(settings.DEFAULT_OUTPUT_WIDTH)
        self.spin_width.setEnabled(False)
        size_layout.addWidget(self.spin_width)
        
        size_layout.addWidget(QLabel("Высота:"))
        self.spin_height = QSpinBox()
        self.spin_height.setRange(100, 10000)
        self.spin_height.setValue(settings.DEFAULT_OUTPUT_HEIGHT)
        self.spin_height.setEnabled(False)
        size_layout.addWidget(self.spin_height)
        
        self.cb_resize.toggled.connect(lambda checked: self.spin_width.setEnabled(checked))
        self.cb_resize.toggled.connect(lambda checked: self.spin_height.setEnabled(checked))
        
        settings_layout.addLayout(size_layout)
        
        # Режим предпросмотра
        self.cb_preview_mode = QCheckBox("Режим предпросмотра (проверка обнаружения)")
        self.cb_preview_mode.setChecked(True)
        self.cb_preview_mode.setToolTip(
            "Включить проверку обнаруженных областей перед обработкой.\n"
            "Вы сможете скорректировать положение экрана для каждого изображения."
        )
        settings_layout.addWidget(self.cb_preview_mode)
        
        # Формат выходного файла
        format_layout = QHBoxLayout()
        format_layout.addWidget(QLabel("Формат:"))
        self.combo_format = QComboBox()
        self.combo_format.addItems(["png", "jpg"])
        format_layout.addWidget(self.combo_format)
        settings_layout.addLayout(format_layout)
        
        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)
        
        # Группа: Выходная директория
        output_group = QGroupBox("Выходная директория")
        output_layout = QVBoxLayout()
        
        self.btn_select_output = QPushButton("Выбрать директорию")
        self.btn_select_output.clicked.connect(self.select_output_directory)
        output_layout.addWidget(self.btn_select_output)
        
        self.label_output = QLabel("Не выбрана")
        self.label_output.setWordWrap(True)
        output_layout.addWidget(self.label_output)
        
        output_group.setLayout(output_layout)
        layout.addWidget(output_group)
        
        # Кнопка обработки
        self.btn_process = QPushButton("Начать обработку")
        self.btn_process.clicked.connect(self.start_processing)
        self.btn_process.setEnabled(False)
        layout.addWidget(self.btn_process)
        
        # Прогресс бар
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        layout.addStretch()
        
        return panel
    
    def create_right_panel(self) -> QWidget:
        """Создание правой панели"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Просмотрщик изображений
        self.image_viewer = ImageViewer()
        layout.addWidget(self.image_viewer)
        
        return panel
    
    def init_ml(self):
        """Инициализация ML моделей"""
        try:
            self.statusBar().showMessage("Загрузка модели...")
            self.detector = ScreenDetector()
            
            if self.detector.is_loaded:
                self.processor = ImageProcessor(self.detector)
                self.statusBar().showMessage("Модель загружена успешно")
            else:
                self.statusBar().showMessage("Модель не загружена")
                QMessageBox.warning(
                    self,
                    "Предупреждение",
                    "Модель детектирования не загружена.\n"
                    "Функция обнаружения экрана будет недоступна."
                )
        except Exception as e:
            logger.error(f"Ошибка инициализации ML: {e}")
            self.statusBar().showMessage("Ошибка загрузки модели")
    
    def add_files(self):
        """Добавление файлов"""
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Выбрать изображения",
            str(Path.home()),
            f"Images (*.{' *.'.join(settings.SUPPORTED_INPUT_FORMATS)})"
        )
        
        if files:
            for file in files:
                file_path = Path(file)
                if file_path not in self.input_files:
                    self.input_files.append(file_path)
                    self.file_list.addItem(file_path.name)
            
            self.update_process_button()
            logger.info(f"Добавлено файлов: {len(files)}")
    
    def add_folder(self):
        """Добавление папки"""
        folder = QFileDialog.getExistingDirectory(
            self,
            "Выбрать папку с изображениями",
            str(Path.home())
        )
        
        if folder:
            files = get_supported_files(Path(folder), recursive=False)
            
            for file_path in files:
                if file_path not in self.input_files:
                    self.input_files.append(file_path)
                    self.file_list.addItem(file_path.name)
            
            self.update_process_button()
            logger.info(f"Добавлено файлов из папки: {len(files)}")
    
    def clear_files(self):
        """Очистка списка файлов"""
        self.input_files.clear()
        self.file_list.clear()
        self.image_viewer.clear()
        self.update_process_button()
    
    def on_file_selected(self, item: QListWidgetItem):
        """Обработка выбора файла в списке"""
        index = self.file_list.row(item)
        if 0 <= index < len(self.input_files):
            file_path = self.input_files[index]
            self.image_viewer.load_image(file_path)
    
    def select_fill_image(self):
        """Выбор изображения для замены экрана"""
        file, _ = QFileDialog.getOpenFileName(
            self,
            "Выбрать изображение для экрана",
            str(Path.home()),
            f"Images (*.{' *.'.join(settings.SUPPORTED_INPUT_FORMATS)})"
        )
        
        if file:
            self.fill_image_path = Path(file)
            logger.info(f"Выбрано изображение для заливки: {self.fill_image_path.name}")
    
    def select_output_directory(self):
        """Выбор выходной директории"""
        folder = QFileDialog.getExistingDirectory(
            self,
            "Выбрать выходную директорию",
            str(Path.home())
        )
        
        if folder:
            self.output_dir = Path(folder)
            self.label_output.setText(str(self.output_dir))
            self.update_process_button()
            logger.info(f"Выходная директория: {self.output_dir}")
    
    def update_process_button(self):
        """Обновление состояния кнопки обработки"""
        can_process = (
            len(self.input_files) > 0 and
            self.output_dir is not None and
            self.processor is not None
        )
        self.btn_process.setEnabled(can_process)
    
    def get_processing_options(self) -> ProcessingOptions:
        """Получение опций обработки из UI"""
        # Определение режима заливки
        if self.rb_black.isChecked():
            fill_mode = ScreenFillMode.BLACK
        elif self.rb_image.isChecked():
            fill_mode = ScreenFillMode.IMAGE
        else:
            fill_mode = ScreenFillMode.NONE
        
        options = ProcessingOptions(
            fill_mode=fill_mode,
            fill_image_path=self.fill_image_path if fill_mode == ScreenFillMode.IMAGE else None,
            use_perspective=self.cb_perspective.isChecked(),
            auto_crop=self.cb_auto_crop.isChecked(),
            resize=self.cb_resize.isChecked(),
            target_width=self.spin_width.value() if self.cb_resize.isChecked() else None,
            target_height=self.spin_height.value() if self.cb_resize.isChecked() else None,
            output_format=self.combo_format.currentText()
        )
        
        return options
    
    def start_processing(self):
        """Запуск обработки"""
        # Валидация
        if not validate_output_directory(self.output_dir):
            QMessageBox.critical(
                self,
                "Ошибка",
                f"Невозможно записать в директорию:\n{self.output_dir}"
            )
            return
        
        options = self.get_processing_options()
        
        # Проверка наличия изображения для заливки
        if options.fill_mode == ScreenFillMode.IMAGE and options.fill_image_path is None:
            QMessageBox.warning(
                self,
                "Предупреждение",
                "Выберите изображение для замены экрана"
            )
            return
        
        # Режим предпросмотра
        if self.cb_preview_mode.isChecked():
            self.start_preview_mode(options)
        else:
            self.start_direct_processing(options)
    
    def start_preview_mode(self, options):
        """Запуск режима предпросмотра"""
        from app.ui.preview_dialog import DetectionPreviewDialog
        
        logger.info("Запуск режима предпросмотра")
        
        # Открытие диалога предпросмотра
        preview_dialog = DetectionPreviewDialog(
            self.input_files,
            self.detector,
            self
        )
        
        # Отмена предпросмотра
        if preview_dialog.exec() != QDialog.Accepted:
            logger.info("Предпросмотр отменён пользователем")
            return
        
        # Получение результатов
        results = preview_dialog.get_results()
        
        # Фильтрация пропущенных файлов
        files_to_process = []
        manual_corrections = {}  # {file_path: (polygon, mask, skip)}

        for file_path, (polygon, mask, skip) in results.items():
            if not skip:
                files_to_process.append(file_path)
                # Всегда в corrections, даже если None (для отслеживания пропуска)
                manual_corrections[file_path] = (polygon, mask, False)  # skip=False т.к. не пропущен
            else:
                # Файл пропущен, но в список для обработки (кадрирование/ресайз)
                files_to_process.append(file_path)
                manual_corrections[file_path] = (None, None, True)  # skip=True
        
        if not files_to_process:
            QMessageBox.information(
                self,
                "Обработка отменена",
                "Все файлы были пропущены"
            )
            return
        
        logger.info(f"К обработке: {len(files_to_process)} файлов")
        logger.info(f"Ручных корректировок: {len(manual_corrections)}")
        
        # Запуск обработки с корректировками
        self.start_processing_with_corrections(
            files_to_process,
            options,
            manual_corrections
        )
    
    def start_direct_processing(self, options):
        """Запуск прямой обработки без предпросмотра"""
        logger.info("Начало прямой обработки")
        
        # Отключение UI
        self.btn_process.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.progress_bar.setMaximum(len(self.input_files))
        
        # Запуск в отдельном потоке
        self.processing_thread = ProcessingThread(
            self.processor,
            self.input_files,
            self.output_dir,
            options
        )
        self.processing_thread.progress.connect(self.on_progress)
        self.processing_thread.finished.connect(self.on_processing_finished)
        self.processing_thread.start()
    
    def start_processing_with_corrections(
        self,
        files: List[Path],
        options,
        corrections: dict
    ):
        """Запуск обработки с ручными корректировками"""
        from app.pipeline.processor import ProcessingResult
        import time
        
        logger.info("Начало обработки с корректировками")
        
        # Отключение UI
        self.btn_process.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.progress_bar.setMaximum(len(files))
        
        # Обработка каждого файла
        results = []
        
        for i, file_path in enumerate(files, 1):
            self.on_progress(i, len(files), file_path.name)
            
            start_time = time.time()
            result = ProcessingResult(input_path=file_path)
            
            try:
                # Загрузка изображения
                from app.core.image_loader import ImageLoader
                image = ImageLoader.load_image(file_path)
                
                if image is None:
                    result.error = "Ошибка загрузки изображения"
                    results.append(result)
                    continue
                
                # Использование корректировок если есть
                if file_path in corrections:
                    polygon, mask, skip = corrections[file_path]
                    
                    if skip:
                        # Файл пропущен
                        result.screen_detected = False
                        polygon = None
                        mask = None
                        logger.info(f"Файл пропущен, экран не обрабатывается: {file_path.name}")
                    else:
                        result.screen_detected = True
                        logger.info(f"Использование ручной корректировки: {file_path.name}")
                else:
                    # Детектирование
                    detection = self.detector.detect_screen(image, options.detect_confidence)
                    
                    if detection is None:
                        result.screen_detected = False
                        
                        if options.fill_mode != ScreenFillMode.NONE:
                            result.error = "Экран не найден"
                            results.append(result)
                            continue
                        
                        mask = None
                        polygon = None
                    else:
                        mask, polygon = detection
                        result.screen_detected = True
                
                # Обработка экрана
                if mask is not None:
                    from app.core.screen_editor import ScreenEditor
                    
                    if options.fill_mode == ScreenFillMode.BLACK:
                        image = ScreenEditor.fill_screen_black(image, mask)
                    
                    elif options.fill_mode == ScreenFillMode.COLOR:
                        image = ScreenEditor.fill_screen_color(image, mask, options.fill_color)
                    
                    elif options.fill_mode == ScreenFillMode.IMAGE:
                        if options.fill_image_path is not None:
                            fill_image = ImageLoader.load_image(options.fill_image_path)
                            
                            if fill_image is not None:
                                if options.use_perspective and polygon is not None and len(polygon.points) >= 4:
                                    image = ScreenEditor.replace_screen_with_perspective(
                                        image, polygon, fill_image
                                    )
                                else:
                                    image = ScreenEditor.replace_screen_with_image(
                                        image, mask, fill_image
                                    )
                
                # Кадрирование
                if options.auto_crop:
                    from app.core.cropper import ImageCropper
                    image = ImageCropper.auto_crop(image, margin=options.crop_margin)
                
                if options.crop_aspect_ratio is not None:
                    from app.core.cropper import ImageCropper
                    image = ImageCropper.crop_to_aspect_ratio(image, options.crop_aspect_ratio)
                
                # Масштабирование
                if options.resize:
                    from app.core.resizer import ImageResizer
                    image = ImageResizer.resize(
                        image,
                        options.target_width,
                        options.target_height,
                        options.maintain_aspect
                    )
                
                # Сохранение
                from app.utils.file_utils import get_output_filename, ensure_directory
                ensure_directory(self.output_dir)
                
                output_filename = get_output_filename(
                    file_path,
                    options.output_format,
                    options.output_suffix
                )
                output_path = self.output_dir / output_filename
                
                success = ImageLoader.save_image(
                    image,
                    output_path,
                    format=options.output_format,
                    quality=options.output_quality
                )
                
                if success:
                    result.success = True
                    result.output_path = output_path
                else:
                    result.error = "Ошибка сохранения"
            
            except Exception as e:
                logger.error(f"Ошибка обработки {file_path}: {e}")
                result.error = str(e)
            
            finally:
                result.processing_time = time.time() - start_time
                results.append(result)
        
        # Завершение
        self.on_processing_finished(results)
    
    def on_progress(self, current: int, total: int, filename: str):
        """Обновление прогресса"""
        self.progress_bar.setValue(current)
        self.statusBar().showMessage(f"Обработка {current}/{total}: {filename}")
    
    def on_processing_finished(self, results: list):
        """Завершение обработки"""
        self.progress_bar.setVisible(False)
        self.btn_process.setEnabled(True)
        
        # Статистика
        report = self.processor.get_processing_report(results)
        
        msg = (
            f"Обработка завершена!\n\n"
            f"Всего: {report['total_files']}\n"
            f"Успешно: {report['successful']}\n"
            f"Ошибок: {report['failed']}\n"
            f"Экран найден: {report['screens_detected']}\n"
            f"Время: {report['total_time']:.2f} сек"
        )
        
        if report['failed'] > 0:
            msg += f"\n\nНе удалось обработать:\n" + "\n".join(report['failed_files'][:5])
            if len(report['failed_files']) > 5:
                msg += f"\n... и ещё {len(report['failed_files']) - 5}"
        
        QMessageBox.information(self, "Результат", msg)
        
        self.statusBar().showMessage("Готов к работе")
        logger.info("Обработка завершена")
    
    def closeEvent(self, event):
        """Обработка закрытия окна"""
        if self.processing_thread and self.processing_thread.isRunning():
            reply = QMessageBox.question(
                self,
                "Подтверждение",
                "Обработка изображений ещё не завершена.\nВы уверены, что хотите выйти?",
                QMessageBox.Yes | QMessageBox.No
            )
            
            if reply == QMessageBox.No:
                event.ignore()
                return
            
            self.processing_thread.terminate()
            self.processing_thread.wait()
        
        event.accept()