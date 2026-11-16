import cv2
import numpy as np

# Orijinal fotoğrafı oku
resim = cv2.imread(str(Path(__file__).resolve().parent.parent / "ornek_fotograflar" / 'bus.jpg'))

# Tekrarlanabilir sonuç için
np.random.seed(42)

# Gaussian gürültü oluştur
gurultu = np.random.normal(
    0, 35, resim.shape
)

# Gürültüyü fotoğrafa ekle
sonuc = resim.astype(np.float32) + gurultu

# Piksel değerlerini sınırla
sonuc = np.clip(sonuc, 0, 255).astype(np.uint8)

# Fotoğrafı kaydet
cv2.imwrite(str(Path(__file__).resolve().parent.parent / "ornek_fotograflar" / 'gurultulu_bus.jpg'), sonuc)

print("Gürültülü fotoğraf oluşturuldu!")
