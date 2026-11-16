
from pathlib import Path
import cv2

# Proje ana klasoru
kok = Path(__file__).resolve().parent.parent

# Giris ve cikis klasorleri
giris = kok / "veri_seti" / "bozulmus" / "gurultulu"
cikis = kok / "veri_seti" / "iyilestirilmis" / "gaussian"

cikis.mkdir(parents=True, exist_ok=True)

if not giris.is_dir():
    raise FileNotFoundError(f"Gurultulu fotograf klasoru bulunamadi: {giris}")

fotograflar = sorted(giris.glob("*.jpg"))

if not fotograflar:
    raise RuntimeError(f"Bu klasorde JPG fotograf bulunamadi: {giris}")

sayac = 0

for foto in fotograflar:
    resim = cv2.imread(str(foto))

    if resim is None:
        print(f"Okunamadi: {foto.name}")
        continue

    # 5x5 Gaussian filtre
    iyilestirilmis = cv2.GaussianBlur(resim, (5, 5), 0)

    hedef = cikis / foto.name

    if cv2.imwrite(str(hedef), iyilestirilmis):
        sayac += 1

print(f"\nGaussian filtre tamamlandi.")
print(f"Islenen fotograf: {sayac}")
print(f"Kaydedilen klasor: {cikis}")
