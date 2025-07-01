import os
import cv2
import numpy as np
from collections import defaultdict
from ultralytics import YOLO

class VehicleTracker:
    """
    Класс для детекции и трекинга транспортных средств на видеопотоке с использованием модели YOLO.

    Детектирует объекты по кадрам видео, определяет их треки и сохраняет траектории движения 
    только для объектов определённых классов транспортных средств (по COCO-классификации):
    - 2: car
    - 5: bus
    - 7: truck

    Атрибуты:
        model (YOLO): Загруженная модель YOLO для инференса.
        vehicle_classes (list): Список id классов объектов, относящихся к транспорту.
    """

    def __init__(self, model_type="yolo11n", device="cpu"):
        """
        Инициализирует VehicleTracker с указанной моделью и устройством для инференса.

        Args:
            model_type (str, optional): Путь к весам YOLO или название модели.
            device (str, optional): Устройство для инференса ("cuda" или "cpu"). По умолчанию "cpu".
        """
        self.model = YOLO(model_type).to(device)
        self.vehicle_classes = [2, 5, 7]

    def track_vehicles(self, video_path):
        """
        Выполняет детекцию и трекинг транспортных средств на видеопотоке.

        Args:
            video_path (str): Путь к видеофайлу.

        Returns:
            dict: Словарь с траекториями объектов.
                  Ключ — track_id (int), 
                  значение — список кортежей (frame_num, (x, y, w, h)) с координатами 
                  центра и размерами бокса на каждом кадре.
        """
        cap = cv2.VideoCapture(video_path)
        vehicle_tracks = defaultdict(list)

        frame_num = 0
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            results = self.model.track(frame, persist=True, verbose=False)
            boxes = results[0].boxes

            if boxes.id is None:
                frame_num += 1
                continue

            classes = boxes.cls.cpu()
            track_ids = boxes.id.int().cpu().tolist()
            boxes_xywh = boxes.xywh.cpu()

            for box, track_id, cls in zip(boxes_xywh, track_ids, classes):
                if cls.item() in self.vehicle_classes:
                    x, y, w, h = box
                    vehicle_tracks[track_id].append((frame_num, (x.item(), y.item(), w.item(), h.item())))

            frame_num += 1

        cap.release()
        return vehicle_tracks
