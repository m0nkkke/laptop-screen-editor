"""
Главная точка входа приложения
"""
import sys
from pathlib import Path
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon
from loguru import logger

# Добавление корневой директории в sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import settings
from app.utils.logger import setup_logger
from app.ui.main_window import MainWindow


def main():
    """Главная функция"""
    # Настройка логирования
    setup_logger()
    logger.info("=" * 60)
    logger.info("Запуск Laptop Screen Editor")
    logger.info("=" * 60)
    
    # Создание приложения
    app = QApplication(sys.argv)
    app.setApplicationName(settings.WINDOW_TITLE)
    app.setOrganizationName("LaptopEditor")
    icon = QIcon("app/assets/icon.ico") 
    app.setWindowIcon(icon)
    
    # Установка стиля
    app.setStyle("Fusion")
    
    # Создание главного окна
    try:
        window = MainWindow()
        window.setWindowIcon(icon)
        window.show()
        
        logger.info("Главное окно отображено")
        
        # Запуск цикла событий
        exit_code = app.exec()
        
        logger.info(f"Приложение завершено с кодом: {exit_code}")
        return exit_code
    
    except Exception as e:
        logger.critical(f"Критическая ошибка при запуске: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())