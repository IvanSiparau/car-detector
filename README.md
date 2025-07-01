# car-detector

# Car Detector — трекинг и детекция остановок транспортных средств на видео

## Описание

Проект для анализа видеопотоков с помощью YOLOv8.  
Выполняет:
- детекцию и трекинг автомобилей, автобусов и грузовиков,
- определение момента остановки транспортного средства относительно заданной линии.

Используется модель YOLOv8 из библиотеки [ultralytics](https://github.com/ultralytics/ultralytics).

---

## Структура проекта

- `tracker.py` — класс `VehicleTracker` для трекинга транспортных средств.
- `stop_detector.py` — класс `StopDetector` для фиксации остановок по трекам.
- `main.py` — запуск анализа видео из консоли.
- `requirements.txt` — список зависимостей.

---

## Установка

1. Клонируйте репозиторий:
```bash
git clone https://github.com/yourusername/car-detector.git
cd car-detector
pip install -r requirements.txt
python main.py path/to/your/video.mp4
```
