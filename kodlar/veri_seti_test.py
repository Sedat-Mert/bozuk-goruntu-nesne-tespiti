
from ultralytics import YOLO
from pathlib import Path
import csv

model = YOLO(str(Path(__file__).resolve().parent.parent / "modeller/yolov8n.pt"))

kok = Path(__file__).resolve().parent.parent / 'veri_seti'
orijinal = kok / "coco128"
etiketler = orijinal / "labels" / "train2017"

gruplar = {
    "Orijinal": orijinal / "images" / "train2017",
    "Karanlik": kok / "bozulmus" / "karanlik",
    "Bulanik": kok / "bozulmus" / "bulanik",
    "Gurultulu": kok / "bozulmus" / "gurultulu",
}

sonuc_klasoru = Path(__file__).resolve().parent.parent / 'sonuclar'
sonuc_klasoru.mkdir(exist_ok=True)

veriler = []

for ad, resim_klasoru in gruplar.items():
    print(f"\nTest ediliyor: {ad}")

    if not resim_klasoru.exists():
        raise FileNotFoundError(resim_klasoru)

    yaml_yolu = sonuc_klasoru / f"{ad.lower()}_test.yaml"

    # Tum gruplar ayni orijinal etiketleri kullanir.
    # YAML icindeki val yolu goruntu klasorudur.
    # Ultralytics etiketleri standart images/labels
    # klasor yapisindan buldugu icin test icin
    # gecici bir veri seti yapisi kuruyoruz.
    test_kok = sonuc_klasoru / "test_verileri" / ad.lower()
    test_images = test_kok / "images" / "val"
    test_labels = test_kok / "labels" / "val"

    test_images.mkdir(parents=True, exist_ok=True)
    test_labels.mkdir(parents=True, exist_ok=True)

    for resim in resim_klasoru.glob("*.jpg"):
        hedef = test_images / resim.name
        if not hedef.exists():
            hedef.symlink_to(resim.resolve())

        etiket = etiketler / f"{resim.stem}.txt"
        if etiket.exists():
            hedef_etiket = test_labels / etiket.name
            if not hedef_etiket.exists():
                hedef_etiket.symlink_to(etiket.resolve())

    yaml_yolu.write_text(
        f"path: {test_kok.resolve()}\n"
        "train: images/val\n"
        "val: images/val\n"
        "names:\n"
        + "\n".join(
            f"  {i}: {isim}"
            for i, isim in model.names.items()
        )
        + "\n",
        encoding="utf-8"
    )

    sonuc = model.val(
        data=str(yaml_yolu.resolve()),
        imgsz=640,
        batch=4,
        device="cpu",
        project=str(sonuc_klasoru.resolve()),
        name=f"test_{ad.lower()}",
        exist_ok=True,
        plots=False
    )

    veriler.append({
        "Yontem": ad,
        "Precision": round(sonuc.box.mp, 4),
        "Recall": round(sonuc.box.mr, 4),
        "mAP50": round(sonuc.box.map50, 4),
        "mAP50-95": round(sonuc.box.map, 4)
    })

with open(
    sonuc_klasoru / "veri_seti_karsilastirma.csv",
    "w",
    newline="",
    encoding="utf-8"
) as dosya:
    yazici = csv.DictWriter(
        dosya,
        fieldnames=[
            "Yontem", "Precision", "Recall",
            "mAP50", "mAP50-95"
        ]
    )
    yazici.writeheader()
    yazici.writerows(veriler)

print("\nTum testler tamamlandi!")
for veri in veriler:
    print(veri)
