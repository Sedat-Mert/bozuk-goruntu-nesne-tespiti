
import cv2
import numpy as np

# Karanlık görüntüyü oku
resim = cv2.imread(str(Path(__file__).resolve().parent.parent / "ornek_fotograflar" / 'karanlik_bus.jpg'))

# Gamma düzeltme
gamma = 0.4

tablo = np.array([
    ((i / 255.0) ** gamma) * 255
    for i in range(256)
]).astype("uint8")

# Filtreyi uygula
sonuc = cv2.LUT(resim, tablo)

# Yeni görüntüyü kaydet
cv2.imwrite(str(Path(__file__).resolve().parent.parent / "ornek_fotograflar" / 'gamma_bus.jpg'), sonuc)

print("Gamma düzeltme tamamlandı!")
