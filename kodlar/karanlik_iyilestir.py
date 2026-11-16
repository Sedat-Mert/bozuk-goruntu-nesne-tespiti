
import cv2
import numpy as np
from pathlib import Path

# Klasör yolları
giris = Path(__file__).resolve().parent.parent / 'veri_seti/bozulmus/karanlik'
cikis = Path(__file__).resolve().parent.parent / 'veri_seti/iyilestirilmis/karanlik'

cikis.mkdir(parents=True, exist_ok=True)

# Gamma düzeltmesi
gamma = 2.0

tablo = np.array([
    ((i / 255.0) ** (1.0 / gamma)) * 255
    for i in range(256)
]).astype("uint8")

sayac = 0

for dosya in sorted(giris.iterdir()):
    if dosya.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
        continue

    fotograf = cv2.imread(str(dosya))

    if fotograf is None:
        print(f"Okunamadi: {dosya.name}")
        continue

    iyilestirilmis = cv2.LUT(fotograf, tablo)

    hedef = cikis / dosya.name
    cv2.imwrite(str(hedef), iyilestirilmis)

    sayac += 1

print(f"{sayac} karanlik fotograf iyilestirildi!")
print(f"Kaydedilen klasor: {cikis}")
