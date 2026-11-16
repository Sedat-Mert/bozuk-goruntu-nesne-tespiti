import cv2

# Bulanık fotoğrafı oku
resim = cv2.imread(str(Path(__file__).resolve().parent.parent / "ornek_fotograflar" / 'bulanik_bus.jpg'))

# Ek bulanıklık oluştur
yumustatilmis = cv2.GaussianBlur(
    resim, (0, 0), 3
)

# Keskinleştirme uygula
keskin = cv2.addWeighted(
    resim, 2.0,
    yumustatilmis, -1.0,
    0
)

# Sonucu kaydet
cv2.imwrite(str(Path(__file__).resolve().parent.parent / "ornek_fotograflar" / 'keskin_bus.jpg'), keskin)

print("Keskinleştirme tamamlandı!")
