
from pathlib import Path
import cv2

# =====================================
# 1. KLASOR YOLLARI
# =====================================

kok = Path(__file__).resolve().parent.parent

giris = kok / "veri_seti" / "bozulmus" / "bulanik"
cikis = kok / "veri_seti" / "iyilestirilmis" / "keskinlestirilmis"

cikis.mkdir(parents=True, exist_ok=True)

# =====================================
# 2. FOTOGRAFLARI KONTROL ET
# =====================================

if not giris.is_dir():
    raise FileNotFoundError(
        f"Bulanik fotograf klasoru bulunamadi: {giris}"
    )

fotograflar = sorted(giris.glob("*.jpg"))

if len(fotograflar) != 128:
    raise RuntimeError(
        f"128 fotograf bekleniyordu, {len(fotograflar)} bulundu."
    )

# =====================================
# 3. UNSHARP MASKING UYGULA
# =====================================

sayac = 0

for foto in fotograflar:

    resim = cv2.imread(str(foto))

    if resim is None:
        print(f"Okunamadi: {foto.name}")
        continue

    # Goruntunun yumusatilmis hali
    yumusak = cv2.GaussianBlur(resim, (5, 5), 1.0)

    # Keskinlestirme
    keskin = cv2.addWeighted(
        resim, 1.5,
        yumusak, -0.5,
        0
    )

    hedef = cikis / foto.name

    if cv2.imwrite(str(hedef), keskin):
        sayac += 1
    else:
        print(f"Kaydedilemedi: {foto.name}")

# =====================================
# 4. SONUC
# =====================================

print("\n================================")
print("BULANIK GORUNTU IYILESTIRME")
print("================================")

print(f"Islenen fotograf: {sayac}")
print(f"Kaydedilen klasor: {cikis}")

if sayac == 128:
    print("Tum fotograflar basariyla iyilestirildi!")
else:
    print("UYARI: Bazi fotograflar islenemedi.")
