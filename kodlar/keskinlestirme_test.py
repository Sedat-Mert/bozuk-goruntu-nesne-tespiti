
from pathlib import Path
from ultralytics import YOLO
import yaml
import csv
import shutil

# 1. Klasor yollari
kok = Path(__file__).resolve().parent.parent

model_yolu = kok / "modeller/yolov8n.pt"
fotograflar = kok / "veri_seti/iyilestirilmis/keskinlestirilmis"
etiketler = kok / "veri_seti/coco128/labels/train2017"

test_klasoru = kok / "veri_seti/keskinlestirme_val"
test_images = test_klasoru / "images"
test_labels = test_klasoru / "labels"

# 2. Dosyalari kontrol et
if not model_yolu.is_file():
    raise FileNotFoundError(f"Model bulunamadi: {model_yolu}")

if not fotograflar.is_dir():
    raise FileNotFoundError(f"Fotograflar bulunamadi: {fotograflar}")

if not etiketler.is_dir():
    raise FileNotFoundError(f"Etiketler bulunamadi: {etiketler}")

dosyalar = sorted(fotograflar.glob("*.jpg"))

if len(dosyalar) != 128:
    raise RuntimeError(
        f"128 fotograf bekleniyordu, {len(dosyalar)} bulundu."
    )

# 3. Test klasorunu hazirla
for klasor in (test_images, test_labels):
    if klasor.exists():
        shutil.rmtree(klasor)
    klasor.mkdir(parents=True)

eksik_etiketler = []

for foto in dosyalar:
    (test_images / foto.name).symlink_to(foto.resolve())

    etiket = etiketler / f"{foto.stem}.txt"
    hedef = test_labels / f"{foto.stem}.txt"

    if etiket.exists():
        hedef.symlink_to(etiket.resolve())
    else:
        eksik_etiketler.append(foto.name)
        hedef.touch()

print(f"Hazirlanan fotograf: {len(dosyalar)}")

if eksik_etiketler:
    print("Eksik etiketler:", eksik_etiketler)
    print("UYARI: Bos etiketler dogrulanmali.")

# 4. Model ve YAML
model = YOLO(str(model_yolu))

yaml_dosyasi = test_klasoru / "data.yaml"

with open(yaml_dosyasi, "w", encoding="utf-8") as f:
    yaml.safe_dump({
        "path": str(test_klasoru),
        "train": "images",
        "val": "images",
        "names": dict(model.names)
    }, f, allow_unicode=True, sort_keys=False)

# 5. YOLOv8 testi
print("\nKeskinlestirme testi basliyor...\n")

sonuc = model.val(
    data=str(yaml_dosyasi),
    imgsz=640,
    batch=4,
    device="cpu",
    project=str(kok / "sonuclar"),
    name="keskinlestirme_val",
    exist_ok=True,
    plots=True
)

# 6. Sonuclari hesapla
yeni = {
    "Precision": float(sonuc.box.mp),
    "Recall": float(sonuc.box.mr),
    "mAP50": float(sonuc.box.map50),
    "mAP50-95": float(sonuc.box.map)
}

# Onceki bulanik goruntu testi
eski = {
    "Precision": 0.5166,
    "Recall": 0.4686,
    "mAP50": 0.5042,
    "mAP50-95": 0.3704
}

# 7. Karsilastir ve CSV kaydet
sonuc_klasoru = kok / "sonuclar"
sonuc_klasoru.mkdir(exist_ok=True)

csv_yolu = sonuc_klasoru / "keskinlestirme_karsilastirma.csv"

print("\n=== KESKINLESTIRME KARSILASTIRMASI ===")

with open(csv_yolu, "w", newline="", encoding="utf-8") as f:
    yazici = csv.writer(f)
    yazici.writerow(["Metrik", "Bulanik", "Keskinlestirilmis", "Fark"])

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
