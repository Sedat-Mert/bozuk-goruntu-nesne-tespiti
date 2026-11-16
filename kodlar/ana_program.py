
from pathlib import Path
from ultralytics import YOLO
import cv2
import argparse
import numpy as np

# =====================================
# 1. PROJE YOLLARI
# =====================================

KOK = Path(__file__).resolve().parent.parent
MODEL_YOLU = KOK / "modeller" / "yolov8n.pt"
SONUC_KLASORU = KOK / "sonuclar" / "uygulama"

SONUC_KLASORU.mkdir(parents=True, exist_ok=True)


# =====================================
# 2. GORUNTU IYILESTIRME
# =====================================

def iyilestir(goruntu, yontem):

    if yontem == "gamma":
        # Karanlik goruntuyu aydinlat
        gamma = 0.6
        tablo = np.array([
            ((i / 255.0) ** gamma) * 255
            for i in range(256)
        ], dtype=np.uint8)

        return cv2.LUT(goruntu, tablo)

    elif yontem == "gaussian":
        # Gurultuyu azalt
        return cv2.GaussianBlur(
            goruntu, (5, 5), 0
        )

    elif yontem == "keskin":
        # Bulanik goruntuyu keskinlestir
        yumusak = cv2.GaussianBlur(
            goruntu, (5, 5), 1.0
        )

        return cv2.addWeighted(
            goruntu, 1.5,
            yumusak, -0.5,
            0
        )

    elif yontem == "yok":
        return goruntu.copy()

    else:
        raise ValueError(
            f"Bilinmeyen yontem: {yontem}"
        )


# =====================================
# 3. NESNE TESPITI
# =====================================

def nesne_tespit(model, goruntu):

    sonuc = model.predict(
        source=goruntu,
        imgsz=640,
        conf=0.25,
        device="cpu",
        verbose=False
    )[0]

    isaretli = sonuc.plot()

    nesneler = []

    for kutu in sonuc.boxes:
        sinif_id = int(kutu.cls.item())
        guven = float(kutu.conf.item())

        nesneler.append({
            "isim": model.names[sinif_id],
            "guven": guven
        })

    return isaretli, nesneler


# =====================================
# 4. ANA PROGRAM
# =====================================

def main():

    parser = argparse.ArgumentParser(
        description="Goruntu iyilestirme ve YOLOv8 nesne tespiti"
    )

    parser.add_argument(
        "--foto",
        required=True,
        help="Fotograf dosyasinin yolu"
    )

    parser.add_argument(
        "--yontem",
        choices=["gamma", "gaussian", "keskin", "yok"],
        default="yok",
        help="Goruntu iyilestirme yontemi"
    )

    args = parser.parse_args()

    foto_yolu = Path(args.foto).expanduser().resolve()

    if not foto_yolu.is_file():
        raise FileNotFoundError(
            f"Fotograf bulunamadi: {foto_yolu}"
        )

    if not MODEL_YOLU.is_file():
        raise FileNotFoundError(
            f"YOLO modeli bulunamadi: {MODEL_YOLU}"
        )

    orijinal = cv2.imread(str(foto_yolu))

    if orijinal is None:
        raise RuntimeError(
            "Fotograf okunamadi."
        )

    print("\nGoruntu iyilestiriliyor...")

    duzeltilmis = iyilestir(
        orijinal, args.yontem
    )

    print("YOLOv8 modeli yukleniyor...")

    model = YOLO(str(MODEL_YOLU))

    print("Iyilestirme oncesi nesne tespiti...")

    once_resim, once_nesneler = nesne_tespit(
        model, orijinal
    )

    print("Iyilestirme sonrasi nesne tespiti...")

    sonra_resim, sonra_nesneler = nesne_tespit(
        model, duzeltilmis
    )

    # =====================================
    # 5. SONUCLARI KAYDET
    # =====================================

    ad = foto_yolu.stem
    yontem = args.yontem

    hedef = SONUC_KLASORU / f"{ad}_{yontem}"
    hedef.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(
        str(hedef / "01_orijinal.jpg"),
        orijinal
    )

    cv2.imwrite(
        str(hedef / "02_iyilestirilmis.jpg"),
        duzeltilmis
    )

    cv2.imwrite(
        str(hedef / "03_once_tespit.jpg"),
        once_resim
    )

    cv2.imwrite(
        str(hedef / "04_sonra_tespit.jpg"),
        sonra_resim
    )

    # Iki sonucu yan yana goster
    yukseklik = 480

    def boyutlandir(resim):
        oran = yukseklik / resim.shape[0]
        genislik = int(resim.shape[1] * oran)
        return cv2.resize(
            resim, (genislik, yukseklik)
        )

    sol = boyutlandir(once_resim)
    sag = boyutlandir(sonra_resim)

    karsilastirma = np.hstack([sol, sag])

    cv2.imwrite(
        str(hedef / "05_karsilastirma.jpg"),
        karsilastirma
    )

    # =====================================
    # 6. TERMINAL SONUCLARI
    # =====================================

    print("\n================================")
    print("NESNE TESPITI SONUCLARI")
    print("================================")

    print(f"\nYontem: {yontem}")

    print(f"\nOnce: {len(once_nesneler)} nesne")

    for nesne in once_nesneler:
        print(
            f"  {nesne['isim']} "
            f"(%{nesne['guven'] * 100:.1f})"
        )

    print(f"\nSonra: {len(sonra_nesneler)} nesne")

    for nesne in sonra_nesneler:
        print(
            f"  {nesne['isim']} "
            f"(%{nesne['guven'] * 100:.1f})"
        )

    print(f"\nSonuclar kaydedildi: {hedef}")
    print("Karsilastirma: 05_karsilastirma.jpg")


if __name__ == "__main__":
    main()
