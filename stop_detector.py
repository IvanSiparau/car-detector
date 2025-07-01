import numpy as np
from collections import defaultdict

class StopDetector:
    """
    Класс для детекции остановок транспортных средств по трекингу на видео.

    Останавливает автомобиль, если его центр в течение заданного количества кадров 
    находится в пределах допустимого отклонения по положению и ниже заданной линии.

    Атрибуты:
        line (tuple): Координаты линии в формате (x1, y1, x2, y2).
        stop_frames (int): Количество подряд кадров, в течение которых движение должно 
                           быть минимальным, чтобы зафиксировать остановку.
        max_movement (float): Максимально допустимое перемещение центра объекта 
                              (в пикселях) между кадрами для фиксации как "стоп".
        results (list): Список обнаруженных остановок в формате (track_id, stop_frame).
    """

    def __init__(self, line_coords, stop_frames=20, max_movement=5):
        """
        Инициализирует StopDetector с заданными параметрами.

        Args:
            line_coords (tuple): Координаты линии в формате (x1, y1, x2, y2).
            stop_frames (int, optional): Количество кадров для фиксации остановки. 
                                         По умолчанию 20.
            max_movement (float, optional): Максимальное движение центра (в пикселях)
                                            между кадрами для фиксации остановки. 
                                            По умолчанию 5.
        """
        self.line = line_coords
        self.stop_frames = stop_frames
        self.max_movement = max_movement
        self.results = []

    def _is_stopped(self, positions):
        """
        Проверяет, есть ли в списке позиций окно из stop_frames кадров, 
        в течение которых центр объекта почти не двигался и оказался ниже линии.

        Args:
            positions (list): Список координат центра объекта [(x, y), ...].

        Returns:
            tuple:
                bool: True, если остановка найдена.
                int or None: Индекс последнего кадра окна остановки, если найдена, иначе None.
        """
        for start_idx in range(len(positions) - self.stop_frames + 1):
            end_idx = start_idx + self.stop_frames
            window = positions[start_idx:end_idx]

            base_pos = window[0]
            all_stopped = True

            for pos in window[1:]:
                if np.linalg.norm(np.array(pos) - np.array(base_pos)) > self.max_movement:
                    all_stopped = False
                    break

            if all_stopped and self._is_below_line(window[-1]):
                return True, end_idx - 1

        return False, None

    def _is_below_line(self, point):
        """
        Проверяет, находится ли точка ниже заданной линии.

        Args:
            point (tuple): Координаты точки (x, y).

        Returns:
            bool: True, если точка ниже линии, иначе False.
        """
        x, y = point
        x1, y1, x2, y2 = self.line

        return (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1) > 0

    def predict(self, vehicle_tracks):
        """
        Анализирует треки транспортных средств и фиксирует моменты их остановки.

        Args:
            vehicle_tracks (dict): Словарь с треками объектов. 
                                   Ключ — track_id (int), 
                                   значение — список кортежей (frame_num, (x, y, w, h)).

        Returns:
            list: Список кортежей с зафиксированными остановками в формате (track_id, stop_frame).
        """
        self.results = []

        for track_id, history in vehicle_tracks.items():
            if len(history) < self.stop_frames:
                continue

            positions = [box[:2] for _, box in history]
            frames = [frame for frame, _ in history]

            stopped, window_end_idx = self._is_stopped(positions)
            if stopped:
                stop_frame = frames[window_end_idx]
                self.results.append((track_id, stop_frame))

        return self.results
