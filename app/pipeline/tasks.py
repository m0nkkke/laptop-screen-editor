"""
Задачи для асинхронной обработки
"""
from typing import List, Callable, Optional
from dataclasses import dataclass
from queue import Queue
from threading import Thread
from loguru import logger


@dataclass
class Task:
    """Задача для обработки"""
    task_id: str
    data: any
    callback: Optional[Callable] = None


class TaskQueue:
    """Очередь задач"""
    
    def __init__(self, num_workers: int = 4):
        """
        Инициализация очереди задач
        
        Args:
            num_workers: Количество рабочих потоков
        """
        self.queue = Queue()
        self.num_workers = num_workers
        self.workers: List[Thread] = []
        self.is_running = False
        
        logger.info(f"Очередь задач инициализирована с {num_workers} воркерами")
    
    def start(self):
        """Запуск воркеров"""
        if self.is_running:
            logger.warning("Очередь уже запущена")
            return
        
        self.is_running = True
        
        for i in range(self.num_workers):
            worker = Thread(target=self._worker, args=(i,), daemon=True)
            worker.start()
            self.workers.append(worker)
        
        logger.info(f"Запущено {self.num_workers} воркеров")
    
    def stop(self):
        """Остановка воркеров"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Сигнал остановки для каждого воркера
        for _ in range(self.num_workers):
            self.queue.put(None)
        
        # Ожидание завершения воркеров
        for worker in self.workers:
            worker.join()
        
        self.workers.clear()
        logger.info("Воркеры остановлены")
    
    def add_task(self, task: Task):
        """Добавление задачи в очередь"""
        if not self.is_running:
            logger.warning("Очередь не запущена, задача не будет обработана")
            return
        
        self.queue.put(task)
        logger.debug(f"Задача добавлена: {task.task_id}")
    
    def _worker(self, worker_id: int):
        """Рабочий поток"""
        logger.debug(f"Воркер {worker_id} запущен")
        
        while self.is_running:
            try:
                # Получение задачи
                task = self.queue.get(timeout=1)
                
                if task is None:  # Сигнал остановки
                    break
                
                logger.debug(f"Воркер {worker_id} обрабатывает задачу: {task.task_id}")
                
                # Выполнение callback
                if task.callback:
                    try:
                        task.callback(task.data)
                    except Exception as e:
                        logger.error(f"Ошибка в callback задачи {task.task_id}: {e}")
                
                self.queue.task_done()
            
            except Exception as e:
                if self.is_running:
                    logger.error(f"Ошибка в воркере {worker_id}: {e}")
        
        logger.debug(f"Воркер {worker_id} завершён")
    
    def wait_completion(self):
        """Ожидание завершения всех задач"""
        self.queue.join()
        logger.info("Все задачи выполнены")
    
    def get_queue_size(self) -> int:
        """Получение размера очереди"""
        return self.queue.qsize()


class ParallelProcessor:
    """Параллельный процессор задач"""
    
    def __init__(self, num_workers: int = 4):
        """
        Инициализация процессора
        
        Args:
            num_workers: Количество параллельных потоков
        """
        self.task_queue = TaskQueue(num_workers)
        self.results = []
    
    def process_batch(
        self,
        items: List[any],
        process_func: Callable,
        callback: Optional[Callable] = None
    ) -> List[any]:
        """
        Пакетная обработка элементов
        
        Args:
            items: Список элементов для обработки
            process_func: Функция обработки одного элемента
            callback: Callback для каждого обработанного элемента
        
        Returns:
            Список результатов
        """
        self.results = []
        
        def task_callback(item):
            result = process_func(item)
            self.results.append(result)
            
            if callback:
                callback(result)
        
        # Запуск воркеров
        self.task_queue.start()
        
        # Добавление задач
        for i, item in enumerate(items):
            task = Task(
                task_id=f"task_{i}",
                data=item,
                callback=task_callback
            )
            self.task_queue.add_task(task)
        
        # Ожидание завершения
        self.task_queue.wait_completion()
        self.task_queue.stop()
        
        return self.results