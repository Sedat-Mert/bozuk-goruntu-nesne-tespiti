import cv2

# Orijinal fotoğrafı oku
resim = cv2.imread(str(Path(__file__).resolve().parent.parent / "ornek_fotograflar" / 'bus.jpg'))

# Fotoğrafı bulanıklaştır
bulanik = cv2.GaussianBlur(resim, (15, 15), 0)

# Bulanık fotoğrafı kaydet
cv2.imwrite(str(Path(__file__).resolve().parent.parent / "ornek_fotograflar" / 'bulanik_bus.jpg'), bulanik)

print("Bulanık fotoğraf oluşturuldu!")
