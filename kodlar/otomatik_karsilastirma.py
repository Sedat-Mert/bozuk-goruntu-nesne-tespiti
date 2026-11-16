
import cv2
import csv
import os
from ultralytics import YOLO

# YOLOv8 modelini yükle
model = YOLO(str(Path(__file__).resolve().parent.parent / "modeller/yolov8n.pt"))

# Test edeceğimiz görüntüler
goruntuler = {
    "Normal": str(Path(__file__).resolve().parent.parent / "ornek_fotograflar" / 'bus.jpg'),
    "Karanlik": str(Path(__file__).resolve().parent.parent / "ornek_fotograflar" / 'karanlik_bus.jpg'),
    "Gamma": str(Path(__file__).resolve().parent.parent / "ornek_fotograflar" / 'gamma_bus.jpg'),
    "Bulanik": str(Path(__file__).resolve().parent.parent / "ornek_fotograflar" / 'bulanik_bus.jpg'),
    "Keskinlestirilmis": str(Path(__file__).resolve().parent.parent / "ornek_fotograflar" / 'keskin_bus.jpg'),
    "Gurultulu": str(Path(__file__).resolve().parent.parent / "ornek_fotograflar" / 'gurultulu_bus.jpg'),
    "Gaussian": str(Path(__file__).resolve().parent.parent / "ornek_fotograflar" / 'temiz_bus.jpg'),
    "Median": str(Path(__file__).resolve().parent.parent / "ornek_fotograflar" / 'median_bus.jpg')
}

# Sonuçların kaydedileceği klasör
os.makedirs("sonuclar", exist_ok=True)

veriler = []

for yontem, dosya in goruntuler.items():

    if not os.path.exists(dosya):
        print(f"Dosya bulunamadi: {dosya}")
        continue

    print(f"\nTest ediliyor: {yontem}")

    # Nesne tespiti yap
    sonuc = model.predict(
        source=dosya,
        conf=0.25,
        verbose=False
    )[0]

    # Tespit edilen nesneleri say
    sayilar = {}

    for kutu in sonuc.boxes:
        sinif_id = int(kutu.cls.item())
        nesne = model.names[sinif_id]

        sayilar[nesne] = sayilar.get(nesne, 0) + 1

    insan = sayilar.get("person", 0)
    otobus = sayilar.get("bus", 0)
    toplam = len(sonuc.boxes)

    print(f"Insan: {insan}")
    print(f"Otobus: {otobus}")
    print(f"Toplam tespit: {toplam}")

    # İşaretlenmiş görüntüyü kaydet
    isaretli = sonuc.plot()

    cv2.imwrite(
        f"sonuclar/{yontem}.jpg",
        isaretli
    )

    # Sonuçları tabloya ekle
    veriler.append([
        yontem,
        insan,
        otobus,
        toplam,
        str(sayilar)
    ])

# CSV dosyasına kaydet
with open(
    "sonuclar/karsilastirma.csv",
    "w",
    newline="",
    encoding="utf-8"
) as dosya:

    yazici = csv.writer(dosya)

    yazici.writerow([
        "Yontem",
        "Insan",
        "Otobus",
        "Toplam",
        "Tum Nesneler"
    ])

    yazici.writerows(veriler)

print("\nTum testler tamamlandi!")
print("Sonuclar sonuclar klasorune kaydedildi.")
