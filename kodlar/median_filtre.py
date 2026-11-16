
import cv2

resim = cv2.imread(str(Path(__file__).resolve().parent.parent / "ornek_fotograflar" / 'gurultulu_bus.jpg'))

if resim is None:
    raise FileNotFoundError("gurultulu_bus.jpg bulunamadı!")

temiz = cv2.medianBlur(resim, 5)

cv2.imwrite(str(Path(__file__).resolve().parent.parent / "ornek_fotograflar" / 'median_bus.jpg'), temiz)

print("Median filtre tamamlandı!")
