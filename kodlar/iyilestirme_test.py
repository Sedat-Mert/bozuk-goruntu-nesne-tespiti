from ultralytics import YOLO
from pathlib import Path
import yaml
import csv

# =====================================
# 1. KLASOR YOLLARI
# =====================================

kok = Path(__file__).resolve().parent.parent

fotograflar = kok / "veri_seti/iyilestirilmis/karanlik"
etiketler = kok / "veri_seti/coco128/labels/train2017"

test_klasoru = kok / "veri_seti/karanlik_val"
test_images = test_klasoru / "images"
test_labels = test_klasoru / "labels"

test_images.mkdir(parents=True, exist_ok=True)
test_labels.mkdir(parents=True, exist_ok=True)

# =====================================
# 2. FOTOGRAFLARI VE ETIKETLERI HAZIRLA
# =====================================

if not fotograflar.exists():
    raise FileNotFoundError(
        f"Fotograf klasoru bulunamadi: {fotograflar}"
    )

if not etiketler.exists():
    raise FileNotFoundError(
        f"Etiket klasoru bulunamadi: {etiketler}"
    )

dosyalar = sorted(fotograflar.glob("*.jpg"))

if len(dosyalar) != 128:
    raise RuntimeError(
        f"128 fotograf bekleniyordu, {len(dosyalar)} bulundu."
    )

sayac = 0
eksik_etiketler = []

for foto in dosyalar:

    etiket = etiketler / (foto.stem + ".txt")

    hedef_foto = test_images / foto.name
    hedef_etiket = test_labels / (foto.stem + ".txt")

    # Fotograf baglantisi
    if hedef_foto.is_symlink():
        if hedef_foto.resolve() != foto.resolve():
            hedef_foto.unlink()

    if not hedef_foto.exists():
        hedef_foto.symlink_to(foto.resolve())

    # Etiket baglantisi
    if etiket.exists():

        if hedef_etiket.is_symlink():
            if hedef_etiket.resolve() != etiket.resolve():
                hedef_etiket.unlink()

        if not hedef_etiket.exists():
            hedef_etiket.symlink_to(etiket.resolve())

    else:
        eksik_etiketler.append(foto.name)

        # Sadece gercekten etiketsiz goruntuler
        # icin bos etiket kullanilmalidir.
        if hedef_etiket.is_symlink():
            hedef_etiket.unlink()

        if not hedef_etiket.exists():
            hedef_etiket.touch()

    sayac += 1

print(f"\n{sayac} fotograf hazirlandi.")

if eksik_etiketler:
    print("\nEksik etiketli fotograflar:")
    for ad in eksik_etiketler:
        print("-", ad)

    print(
        "\nUYARI: Bu fotograflarin gercekten "
        "nesne etiketi icermedigini kontrol edin."
    )

# =====================================
# 3. COCO SINIF ISIMLERINI AL
# =====================================

model = YOLO(str(kok / "modeller/yolov8n.pt"))

# Modelin 80 COCO sinifini kullan
siniflar = dict(model.names)

# =====================================
# 4. YAML DOSYASI OLUSTUR
# =====================================

yaml_dosyasi = test_klasoru / "data.yaml"

veri = {
    "path": str(test_klasoru),
    "train": "images",
    "val": "images",
    "names": siniflar
}

with open(yaml_dosyasi, "w", encoding="utf-8") as f:
    yaml.safe_dump(
        veri,
        f,
        allow_unicode=True,
        sort_keys=False
    )

print("Veri seti YAML dosyasi olusturuldu.")

# =====================================
# 5. YOLOV8 DOGRULAMA TESTI
# =====================================

print("\nGamma iyilestirme testi basliyor...\n")

sonuc = model.val(
    data=str(yaml_dosyasi),
    imgsz=640,
    batch=4,
    device="cpu",
    project=str(kok / "sonuclar"),
    name="karanlik_gamma_val",
    exist_ok=True,
    plots=True
)

# =====================================
# 6. SONUCLARI HESAPLA
# =====================================

precision = float(sonuc.box.mp)
recall = float(sonuc.box.mr)
map50 = float(sonuc.box.map50)
map5095 = float(sonuc.box.map)

print("\n================================")
print("GAMMA DUZELTMESI SONUCLARI")
print("================================")

print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"mAP50     : {map50:.4f}")
print(f"mAP50-95  : {map5095:.4f}")

# =====================================
# 7. ESKI SONUCLARLA KARSILASTIR
# =====================================

eski = {
    "Precision": 0.6091,
    "Recall": 0.5360,
    "mAP50": 0.5883,
    "mAP50-95": 0.4410
}

yeni = {
    "Precision": precision,
    "Recall": recall,
    "mAP50": map50,
    "mAP50-95": map5095
}

print("\nONCE / SONRA KARSILASTIRMASI")
print("--------------------------------")

for metrik in eski:
    fark = yeni[metrik] - eski[metrik]

    print(
        f"{metrik}: "
        f"{eski[metrik]:.4f} -> "
        f"{yeni[metrik]:.4f} "
        f"(Fark: {fark:+.4f})"
    )

# =====================================
# 8. CSV OLARAK KAYDET
# =====================================

sonuc_klasoru = kok / "sonuclar"
sonuc_klasoru.mkdir(exist_ok=True)

csv_yolu = sonuc_klasoru / "gamma_karsilastirma.csv"

with open(
    csv_yolu,
    "w",
    newline="",
    encoding="utf-8"
) as dosya:

    yazici = csv.writer(dosya)

    yazici.writerow([
        "Metrik",
        "Once",
        "Sonra",
        "Fark"
    ])

    for metrik in eski:
        yazici.writerow([
            metrik,
            eski[metrik],
            yeni[metrik],
            yeni[metrik] - eski[metrik]
        ])

print("\nTest tamamlandi!")
print(f"CSV kaydedildi: {csv_yolu}")