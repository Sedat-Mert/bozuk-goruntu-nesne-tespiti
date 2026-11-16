
from ultralytics import YOLO
from pathlib import Path

model = YOLO(str(Path(__file__).resolve().parent.parent / "modeller/yolov8n.pt"))

klasor = Path(__file__).resolve().parent.parent / 'veri_seti/iyilestirilmis/karanlik'

sonuclar = model.predict(
    source=str(klasor),
    imgsz=640,
    conf=0.25,
    device="cpu",
    save=True,
    project="sonuclar",
    name="karanlik_iyilestirilmis"
)

print(f"Test tamamlandi! {len(sonuclar)} fotograf islendi.")
