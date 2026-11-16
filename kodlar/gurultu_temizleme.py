
import cv2

# Gürültülü fotoğrafı oku
resim = cv2.imread(str(Path(__file__).resolve().parent.parent / "ornek_fotograflar" / 'gurultulu_bus.jpg'))

# Gaussian filtre uygula
temiz = cv2.GaussianBlur(
    resim, (5, 5), 0
)

# Temizlenmiş fotoğrafı kaydet
cv2.imwrite(str(Path(__file__).resolve().parent.parent / "ornek_fotograflar" / 'temiz_bus.jpg'), temiz)

print("Gürültü temizleme tamamlandı!")
