
import cv2
import numpy as np
from pathlib import Path

# Klasörler
kaynak = Path(__file__).resolve().parent.parent / 'veri_seti/coco128/images/train2017'
hedef = Path(__file__).resolve().parent.parent / 'veri_seti/bozulmus'

turler = ["karanlik", "bulanik", "gurultulu"]

for tur in turler:
    (hedef / tur).mkdir(parents=True, exist_ok=True)

if not kaynak.exists():
    raise FileNotFoundError(f"Veri seti bulunamadi: {kaynak}")

rng = np.random.default_rng(42)
sayac = 0

for dosya in sorted(kaynak.glob("*.jpg")):
    resim = cv2.imread(str(dosya))

    if resim is None:
        continue

    # 1. Karanlik fotograf
    karanlik = cv2.convertScaleAbs(
        resim, alpha=0.4, beta=0
    )

    # 2. Bulanik fotograf
    bulanik = cv2.GaussianBlur(resim, (11, 11), 0)

    # 3. Gurultulu fotograf
    gurultu = rng.normal(0, 25, resim.shape)
    gurultulu = np.clip(
        resim.astype(np.float32) + gurultu,
        0, 255
    ).astype(np.uint8)

    cv2.imwrite(str(hedef / "karanlik" / dosya.name), karanlik)
    cv2.imwrite(str(hedef / "bulanik" / dosya.name), bulanik)
    cv2.imwrite(str(hedef / "gurultulu" / dosya.name), gurultulu)

    sayac += 1

print(f"{sayac} orijinal fotograf islendi.")
print(f"{sayac * 3} bozulmus fotograf olusturuldu.")
print("Islem tamamlandi!")
