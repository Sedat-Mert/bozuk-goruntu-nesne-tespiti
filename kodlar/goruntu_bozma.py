import cv2
import numpy as np

# Orijinal fotoğrafı oku
resim = cv2.imread(str(Path(__file__).resolve().parent.parent / "ornek_fotograflar" / 'bus.jpg'))

# Fotoğrafı karart
karanlik = (resim.astype(np.float32) / 255.0) ** 4
karanlik = (karanlik * 255).astype(np.uint8)

# Karanlık fotoğrafı kaydet
cv2.imwrite(str(Path(__file__).resolve().parent.parent / "ornek_fotograflar" / 'karanlik_bus.jpg'), karanlik)

print("Karanlık fotoğraf oluşturuldu!")
