
from ultralytics import YOLO
from pathlib import Path
# Hazır eğitilmiş YOLOv8 modelini yükle
model = YOLO(str(Path(__file__).resolve().parent.parent / "modeller/yolov8n.pt"))

# Örnek fotoğraf üzerinde nesne tespiti yap
results = model.predict(
    source=str(Path(__file__).resolve().parent.parent / "ornek_fotograflar/bus.jpg"),
    save=True
)

print("Nesne tespiti tamamlandı!")
