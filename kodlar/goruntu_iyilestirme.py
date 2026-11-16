
import cv2

# Karanlık fotoğrafı oku
resim = cv2.imread(str(Path(__file__).resolve().parent.parent / "ornek_fotograflar" / 'karanlik_bus.jpg'))

# LAB renk uzayına dönüştür
lab = cv2.cvtColor(resim, cv2.COLOR_BGR2LAB)

# Kanalları ayır
l, a, b = cv2.split(lab)

# CLAHE filtresini oluştur
clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8)
)

# Parlaklık kanalını iyileştir
l = clahe.apply(l)

# Kanalları birleştir
iyilestirilmis = cv2.merge((l, a, b))

# Tekrar BGR formatına dönüştür
sonuc = cv2.cvtColor(
    iyilestirilmis,
    cv2.COLOR_LAB2BGR
)

# Fotoğrafı kaydet
cv2.imwrite(str(Path(__file__).resolve().parent.parent / "ornek_fotograflar" / 'iyilestirilmis_bus.jpg'), sonuc)

print("Görüntü iyileştirildi!")
