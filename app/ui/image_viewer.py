"""
Виджет для просмотра и редактирования изображений
"""
from pathlib import Path
from typing import Optional
import numpy as np
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QGraphicsView,
    QGraphicsScene, QGraphicsPixmapItem, QGraphicsPolygonItem
)
from PySide6.QtCore import Qt, QRectF, QPointF
from PySide6.QtGui import QPixmap, QImage, QPen, QBrush, QColor, QPolygonF, QPainter
from loguru import logger

from app.core.image_loader import ImageLoader
from app.utils.geometry import Polygon


class ImageViewer(QWidget):
    """Виджет для просмотра изображений"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.current_image: Optional[np.ndarray] = None
        self.current_path: Optional[Path] = None
        self.current_polygon: Optional[Polygon] = None
        
        self.init_ui()
    
    def init_ui(self):
        """Инициализация UI"""
        layout = QVBoxLayout(self)
        
        # Графический вьюер
        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)
        self.view.setRenderHints(
            self.view.renderHints() | QPainter.RenderHint.Antialiasing
        )
        self.view.setDragMode(QGraphicsView.ScrollHandDrag)
        
        layout.addWidget(self.view)
        
        # Элементы сцены
        self.pixmap_item: Optional[QGraphicsPixmapItem] = None
        self.polygon_item: Optional[QGraphicsPolygonItem] = None
    
    def load_image(self, file_path: Path) -> bool:
        """
        Загрузка изображения
        
        Args:
            file_path: Путь к файлу
        
        Returns:
            True если успешно загружено
        """
        try:
            logger.debug(f"Загрузка изображения в viewer: {file_path}")
            
            # Загрузка изображения
            image = ImageLoader.load_image(file_path)
            if image is None:
                return False
            
            # Изменение размера для предпросмотра
            preview_image = ImageLoader.resize_for_preview(image)
            
            # Отображение
            self.display_image(preview_image)
            
            self.current_image = image
            self.current_path = file_path
            
            return True
        
        except Exception as e:
            logger.error(f"Ошибка загрузки изображения в viewer: {e}")
            return False
    
    def display_image(self, image: np.ndarray):
        """
        Отображение изображения
        
        Args:
            image: Изображение в формате numpy array (RGB)
        """
        # Полная очистка сцены
        self.scene.clear()
        self.pixmap_item = None
        self.polygon_item = None
        
        # Конвертация в QPixmap
        h, w, ch = image.shape
        bytes_per_line = ch * w
        
        q_image = QImage(
            image.data,
            w, h,
            bytes_per_line,
            QImage.Format_RGB888
        )
        
        pixmap = QPixmap.fromImage(q_image)
        
        # Добавление на сцену
        self.pixmap_item = self.scene.addPixmap(pixmap)
        self.view.fitInView(self.pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)
    
    def display_polygon(self, polygon: Polygon, color: QColor = QColor(0, 255, 0)):
        """
        Отображение полигона на изображении
        
        Args:
            polygon: Полигон для отображения
            color: Цвет полигона
        """
        if self.pixmap_item is None:
            return
        
        # Удаление предыдущего полигона
        if self.polygon_item:
            self.scene.removeItem(self.polygon_item)
        
        # Создание QPolygonF
        points = [QPointF(p.x, p.y) for p in polygon.points]
        q_polygon = QPolygonF(points)
        
        # Стиль
        pen = QPen(color, 2)
        brush = QBrush(QColor(color.red(), color.green(), color.blue(), 50))
        
        # Добавление на сцену
        self.polygon_item = self.scene.addPolygon(q_polygon, pen, brush)
        self.current_polygon = polygon
    
    def clear_polygon(self):
        """Удаление полигона"""
        if self.polygon_item:
            self.scene.removeItem(self.polygon_item)
            self.polygon_item = None
            self.current_polygon = None
    
    def clear(self):
        """Очистка вьюера"""
        self.scene.clear()
        self.pixmap_item = None
        self.polygon_item = None
        self.current_image = None
        self.current_path = None
        self.current_polygon = None
    
    def zoom_in(self):
        """Приблизить"""
        self.view.scale(1.2, 1.2)
    
    def zoom_out(self):
        """Отдалить"""
        self.view.scale(1 / 1.2, 1 / 1.2)
    
    def fit_to_view(self):
        """Подогнать под размер окна"""
        if self.pixmap_item:
            self.view.fitInView(self.pixmap_item, Qt.KeepAspectRatio)
    
    def get_current_image(self) -> Optional[np.ndarray]:
        """Получение текущего изображения"""
        return self.current_image
    
    def get_current_polygon(self) -> Optional[Polygon]:
        """Получение текущего полигона"""
        return self.current_polygon


class EditableGraphicsView(QGraphicsView):
    """Кастомный QGraphicsView с обработкой событий мыши для редактирования"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_viewer = parent
        self.dragging_point = None
        self.panning = False
        self.pan_start_pos = None
        
        # Настройка zoom
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
    
    def wheelEvent(self, event):
        """Обработка колеса мыши для zoom (с Ctrl) или скролла"""
        if self.parent_viewer and self.parent_viewer.editing_mode:
            # Зажат ли Ctrl
            modifiers = event.modifiers()
            
            if modifiers & Qt.KeyboardModifier.ControlModifier:
                # CTRL зажат - zoom
                zoom_in_factor = 1.15
                zoom_out_factor = 1 / zoom_in_factor
                
                angle_delta = event.angleDelta().y()
                
                if angle_delta > 0:
                    zoom_factor = zoom_in_factor
                else:
                    zoom_factor = zoom_out_factor
                
                self.scale(zoom_factor, zoom_factor)
                event.accept()
            else:
                # CTRL не зажат - обычный скролл по вертикали
                super().wheelEvent(event)
        else:
            super().wheelEvent(event)
    
    def mousePressEvent(self, event):
            """Обработка нажатия мыши"""
            if event.button() == Qt.MouseButton.MiddleButton:
                self.panning = True
                self.pan_start_pos = event.pos()
                self.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)
                event.accept()
                return
            
            if not self.parent_viewer or not self.parent_viewer.editing_mode or not self.parent_viewer.current_polygon:
                return super().mousePressEvent(event)
            
            scene_pos = self.mapToScene(event.pos())
            
            min_dist = float('inf')
            closest_idx = None
            
            for i, point in enumerate(self.parent_viewer.current_polygon.points):
                dist = ((point.x - scene_pos.x()) ** 2 + (point.y - scene_pos.y()) ** 2) ** 0.5
                if dist < min_dist and dist < 15:
                    min_dist = dist
                    closest_idx = i
            
            if closest_idx is not None:
                if event.button() == Qt.MouseButton.RightButton:
                    # Нельзя удалять если точек меньше 4 (минимум для полигона)
                    if len(self.parent_viewer.current_polygon.points) <= 3:
                        from PySide6.QtWidgets import QToolTip
                        QToolTip.showText(
                            event.globalPosition().toPoint(),
                            "Нельзя удалить точку: минимум 3 точки",
                            self
                        )
                        event.accept()
                        return
                    
                    del self.parent_viewer.current_polygon.points[closest_idx]
                    self.parent_viewer.display_polygon(self.parent_viewer.current_polygon)
                    logger.info(f"Удалена точка {closest_idx}, осталось точек: {len(self.parent_viewer.current_polygon.points)}")
                    event.accept()
                elif event.button() == Qt.MouseButton.LeftButton:
                    self.dragging_point = closest_idx
                    self.parent_viewer.dragging_point = closest_idx
                    self.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)
                    event.accept()
            else:
                super().mousePressEvent(event)
    
    def mouseMoveEvent(self, event):
        """Обработка перемещения мыши"""
        # Перемещение view средней кнопкой
        if self.panning and self.pan_start_pos is not None:
            delta = event.pos() - self.pan_start_pos
            self.pan_start_pos = event.pos()
            
            # Скроллбары
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - delta.x()
            )
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - delta.y()
            )
            event.accept()
            return
        
        if not self.parent_viewer or not self.parent_viewer.editing_mode:
            return super().mouseMoveEvent(event)
        
        scene_pos = self.mapToScene(event.pos())
        
        # Изменение курсора при наведении
        if self.dragging_point is None:
            near_point = False
            if self.parent_viewer.current_polygon:
                for point in self.parent_viewer.current_polygon.points:
                    dist = ((point.x - scene_pos.x()) ** 2 + (point.y - scene_pos.y()) ** 2) ** 0.5
                    if dist < 15:
                        near_point = True
                        break
            
            if near_point:
                self.viewport().setCursor(Qt.CursorShape.OpenHandCursor)
            else:
                self.viewport().setCursor(Qt.CursorShape.ArrowCursor)
        
        # Перемещение точки
        if self.dragging_point is not None:
            if self.parent_viewer.pixmap_item:
                rect = self.parent_viewer.pixmap_item.boundingRect()
                scene_pos.setX(max(rect.left(), min(scene_pos.x(), rect.right())))
                scene_pos.setY(max(rect.top(), min(scene_pos.y(), rect.bottom())))
            
            point = self.parent_viewer.current_polygon.points[self.dragging_point]
            point.x = scene_pos.x()
            point.y = scene_pos.y()
            
            self.parent_viewer.display_polygon(self.parent_viewer.current_polygon)
            event.accept()
        else:
            super().mouseMoveEvent(event)
    
    def mouseReleaseEvent(self, event):
        """Обработка отпускания мыши"""
        # Окончание перемещения средней кнопкой
        if event.button() == Qt.MouseButton.MiddleButton and self.panning:
            self.panning = False
            self.pan_start_pos = None
            self.viewport().setCursor(Qt.CursorShape.ArrowCursor)
            event.accept()
            return
        
        if self.dragging_point is not None:
            self.dragging_point = None
            if self.parent_viewer:
                self.parent_viewer.dragging_point = None
            self.viewport().setCursor(Qt.CursorShape.ArrowCursor)
            event.accept()
        else:
            super().mouseReleaseEvent(event)


class InteractiveImageViewer(ImageViewer):
    """
    Интерактивный вьюер с возможностью редактирования полигона
    """
    
    def __init__(self, parent=None):
        # Инициализация переменных ДО вызова super().__init__
        self.editing_mode = False
        self.dragging_point = None
        self.point_items = []
        self.scale_factor = 1.0
        
        super().__init__(parent)
        
        # Замена view на кастомный
        old_view = self.view
        layout = self.layout()
        layout.removeWidget(old_view)
        
        # Новый view ДО удаления старого
        self.view = EditableGraphicsView(self)
        self.view.setScene(self.scene)
        self.view.setRenderHints(
            self.view.renderHints() | QPainter.RenderHint.Antialiasing
        )
        layout.insertWidget(1, self.view)
        
        # Удаляем старый view
        old_view.hide()
        old_view.setParent(None)
        old_view.deleteLater()
    
    def enable_editing(self, enable: bool = True):
        """Включение/отключение режима редактирования"""
        self.editing_mode = enable
        
        if enable:
            self.view.setDragMode(QGraphicsView.NoDrag)
            if self.current_polygon:
                self._draw_control_points()
        else:
            self.view.setDragMode(QGraphicsView.ScrollHandDrag)
            self._clear_control_points()
    
    def display_polygon(self, polygon: Polygon, color: QColor = QColor(0, 255, 0)):
        """Отображение полигона с точками управления"""
        self._clear_control_points()
        
        # Старый полигон
        if self.polygon_item and self.polygon_item.scene():
            self.scene.removeItem(self.polygon_item)
            self.polygon_item = None
        
        # Новый полигон
        points = [QPointF(p.x, p.y) for p in polygon.points]
        q_polygon = QPolygonF(points)
        
        # Стиль полигона
        pen = QPen(QColor(color.red(), color.green(), color.blue(), 100), 0.5)
        brush = QBrush(QColor(color.red(), color.green(), color.blue(), 30))
        
        self.polygon_item = self.scene.addPolygon(q_polygon, pen, brush)
        self.current_polygon = polygon
        
        # Новые точки
        if self.editing_mode:
            self._draw_control_points()
    
    def _draw_control_points(self):
        """Отрисовка управляющих точек"""
        from PySide6.QtWidgets import QGraphicsEllipseItem
        
        self._clear_control_points()
        
        if self.current_polygon is None:
            return
        
        for i, point in enumerate(self.current_polygon.points):
            size = 8  
            ellipse = QGraphicsEllipseItem(
                point.x - size/2,
                point.y - size/2,
                size,
                size
            )
            
            ellipse.setBrush(QBrush(QColor(255, 255, 0, 120)))
            ellipse.setPen(QPen(QColor(255, 0, 0), 1))
            ellipse.setZValue(100)
            
            self.scene.addItem(ellipse)
            self.point_items.append((i, ellipse))
    
    def _clear_control_points(self):
        """Удаление управляющих точек"""
        for idx, item in list(self.point_items):
            try:
                if item.scene() is not None:
                    self.scene.removeItem(item)
            except RuntimeError:
                pass  # Объект уже удалён
        self.point_items.clear()
    
    def clear(self):
        """Очистка вьюера"""
        self._clear_control_points()
        
        self.view.resetTransform()
        
        super().clear()
    
    def get_edited_polygon(self) -> Optional[Polygon]:
        """Получение отредактированного полигона с учётом масштаба"""
        if self.current_polygon is None:
            return None
        
        if self.scale_factor != 1.0:
            scaled_points = [
                (p.x / self.scale_factor, p.y / self.scale_factor) 
                for p in self.current_polygon.points
            ]
            return Polygon(scaled_points)
        
        return self.current_polygon
    
    def set_scale_factor(self, scale: float):
        """Установка коэффициента масштабирования для превью"""
        self.scale_factor = scale
    
    def zoom_in(self):
        """Приблизить"""
        self.view.scale(1.2, 1.2)
    
    def zoom_out(self):
        """Отдалить"""
        self.view.scale(1 / 1.2, 1 / 1.2)
    
    def reset_zoom(self):
        """Сбросить zoom"""
        if self.pixmap_item:
            self.view.fitInView(self.pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)
    def showEvent(self, event):
        super().showEvent(event)
        self.reset_zoom()
