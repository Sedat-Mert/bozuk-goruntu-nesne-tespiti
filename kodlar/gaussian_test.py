
from pathlib import Path
from ultralytics import YOLO
import yaml
import csv
import shutil

kok = Path(__file__).resolve().parent.parent

model_yolu = kok / "modeller" / "yolov8n.pt"
fotograflar = kok / "veri_seti/iyilestirilmis/gaussian"
etiketler = kok / "veri_seti/coco128/labels/train2017"

test_klasoru = kok / "veri_seti/gaussian_val"
test_images = test_klasoru / "images"
test_labels = test_klasoru / "labels"

if not model_yolu.is_file():
    raise FileNotFoundError(f"Model bulunamadi: {model_yolu}")

if not fotograflar.is_dir() or not etiketler.is_dir():
    raise FileNotFoundError("Fotograf veya etiket klasoru bulunamadi.")

dosyalar = sorted(fotograflar.glob("*.jpg"))

if len(dosyalar) != 128:
    raise RuntimeError(f"128 fotograf bekleniyordu, {len(dosyalar)} bulundu.")

# Eski test dosyalarini temizle
for klasor in (test_images, test_labels):
    if klasor.exists():
        shutil.rmtree(klasor)
    klasor.mkdir(parents=True)

eksik_etiketler = []

for foto in dosyalar:
    hedef_foto = test_images / foto.name
    hedef_foto.symlink_to(foto.resolve())

    etiket = etiketler / f"{foto.stem}.txt"
    hedef_etiket = test_labels / f"{foto.stem}.txt"

    if etiket.exists():
        hedef_etiket.symlink_to(etiket.resolve())
    else:
        eksik_etiketler.append(foto.name)
        hedef_etiket.touch()

print(f"\nHazirlanan fotograf: {len(dosyalar)}")

if eksik_etiketler:
    print("Eksik etiketler:", eksik_etiketler)
    print("UYARI: Bos etiketlerin dogrulugu kontrol edilmelidir.")

model = YOLO(str(model_yolu))

yaml_dosyasi = test_klasoru / "data.yaml"

with open(yaml_dosyasi, "w", encoding="utf-8") as f:
    yaml.safe_dump({
        "path": str(test_klasoru),
        "train": "images",
        "val": "images",
        "names": dict(model.names)
    }, f, allow_unicode=True, sort_keys=False)

print("\nGaussian YOLOv8 testi basliyor...\n")

sonuc = model.val(
    data=str(yaml_dosyasi),
    imgsz=640,
    batch=4,
    device="cpu",
    project=str(kok / "sonuclar"),
    name="gaussian_val",
    exist_ok=True,
    plots=True
)

yeni = {
    "Precision": float(sonuc.box.mp),
    "Recall": float(sonuc.box.mr),
    "mAP50": float(sonuc.box.map50),
    "mAP50-95": float(sonuc.box.map)
}

# Daha once olctugumuz gurultulu goruntu sonuclari
eski = {
    "Precision": 0.6691,
    "Recall": 0.3183,
    "mAP50": 0.3945,
    "mAP50-95": 0.2803
}

print("\n=== GAUSSIAN FILTRE KARSILASTIRMASI ===")

sonuc_klasoru = kok / "sonuclar"
sonuc_klasoru.mkdir(exist_ok=True)

csv_yolu = sonuc_klasoru / "gaussian_karsilastirma.csv"

with open(csv_yolu, "w", newline="", encoding="utf-8") as f:
    yazici = csv.writer(f)
    yazici.writerow(["Metrik", "Gurultulu", "Gaussian", "Fark"])

    for metrik in eski:
        fark = yeni[metrik] - eski[metrik]
        print(
            f"{metrik}: {eski[metrik]:.4f} -> "
            f"{yeni[metrik]:.4f} ({fark:+.4f})"
        )
        yazici.writerow([
            metrik,
            eski[metrik],
            yeni[metrik],
            fark
        ])

print("\nTest tamamlandi!")
print(f"CSV kaydedildi: {csv_yolu}")
