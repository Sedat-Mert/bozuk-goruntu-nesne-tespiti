
from ultralytics import YOLO
from pathlib import Path

# YOLOv8 modelini yükle
model = YOLO(str(Path(__file__).resolve().parent.parent / "modeller/yolov8n.pt"))

# COCO128 veri setini indir ve değerlendir
sonuclar = model.val(
    data="coco128.yaml",
    imgsz=640,
    batch=4,
    device="cpu",
    project="sonuclar",
    name="ilk_veri_seti_testi"
)

print("\nVeri seti testi tamamlandı!")
print(f"Precision: {sonuclar.box.mp:.4f}")
print(f"Recall: {sonuclar.box.mr:.4f}")
print(f"mAP50: {sonuclar.box.map50:.4f}")
print(f"mAP50-95: {sonuclar.box.map:.4f}")
