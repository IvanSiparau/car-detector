import argparse
from tracker import VehicleTracker
from stop_detector import StopDetector


def main(video_path):
    """
    Основная функция трекинга и детекции остановок на видео.

    Args:
        video_path (str): Путь к видеофайлу.
    """
    print(f"Загружаем видео: {video_path}")

    tracker = VehicleTracker()
    stop_detector = StopDetector(line_coords=(243,147,513,147), stop_frames=20, max_movement=5)

    print("Выполняется трекинг объектов...")
    vehicle_tracks = tracker.track_vehicles(video_path)
    print(f"Найдено {len(vehicle_tracks)} треков.")

    results = stop_detector.predict(vehicle_tracks)

    print("\nРезультаты:")
    for track_id, stop_frame in results:
        print(f"Track ID {track_id} остановился на кадре {stop_frame}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vehicle Tracking & Stop Detection")
    parser.add_argument("video_path", type=str, help="Путь до видеофайла для анализа")

    args = parser.parse_args()

    main(args.video_path)
