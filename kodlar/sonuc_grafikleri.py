
from pathlib import Path
import csv
import matplotlib.pyplot as plt

# =====================================
# 1. KLASOR YOLLARI
# =====================================

kok = Path(__file__).resolve().parent.parent
sonuclar = kok / "sonuclar"
grafikler = sonuclar / "grafikler"
grafikler.mkdir(parents=True, exist_ok=True)

deneyler = {
    "Gamma": "gamma_karsilastirma.csv",
    "Gaussian": "gaussian_karsilastirma.csv",
    "Keskinlestirme": "keskinlestirme_karsilastirma.csv"
}

metrikler = ["Precision", "Recall", "mAP50", "mAP50-95"]
veriler = {}

# =====================================
# 2. CSV DOSYALARINI OKU
# =====================================

for deney, dosya_adi in deneyler.items():
    dosya = sonuclar / dosya_adi

    if not dosya.exists():
        raise FileNotFoundError(
            f"Sonuc dosyasi bulunamadi: {dosya}"
        )

    veriler[deney] = {}

    with open(dosya, "r", encoding="utf-8-sig", newline="") as f:
        okuyucu = csv.DictReader(f)

        for satir in okuyucu:
            metrik = satir["Metrik"]

            if metrik not in metrikler:
                continue

            veriler[deney][metrik] = {
                "once": float(satir["Once"])
                    if "Once" in satir
                    else float(satir["Gurultulu"])
                    if "Gurultulu" in satir
                    else float(satir["Bulanik"]),

                "sonra": float(satir["Sonra"])
                    if "Sonra" in satir
                    else float(satir["Gaussian"])
                    if "Gaussian" in satir
                    else float(satir["Keskinlestirilmis"])
            }

# =====================================
# 3. HER DENEY ICIN GRAFIK
# =====================================

for deney, sonuc in veriler.items():
    once = [sonuc[m]["once"] * 100 for m in metrikler]
    sonra = [sonuc[m]["sonra"] * 100 for m in metrikler]

    x = list(range(len(metrikler)))
    genislik = 0.35

    plt.figure(figsize=(10, 6))

    plt.bar(
        [i - genislik / 2 for i in x],
        once,
        width=genislik,
        label="Iyilestirme oncesi"
    )

    plt.bar(
        [i + genislik / 2 for i in x],
        sonra,
        width=genislik,
        label="Iyilestirme sonrasi"
    )

    plt.xticks(x, metrikler)
    plt.ylim(0, 100)
    plt.ylabel("Basari (%)")
    plt.title(f"{deney} - YOLOv8 Performans Karsilastirmasi")
    plt.legend()
    plt.grid(axis="y", alpha=0.2)
    plt.tight_layout()

    hedef = grafikler / f"{deney.lower()}_grafik.png"
    plt.savefig(hedef, dpi=200)
    plt.close()

    print(f"Grafik kaydedildi: {hedef.name}")

# =====================================
# 4. GENEL mAP50 KARSILASTIRMASI
# =====================================

isimler = list(veriler.keys())
farklar = [
    (veriler[ad]["mAP50"]["sonra"]
     - veriler[ad]["mAP50"]["once"]) * 100
    for ad in isimler
]

plt.figure(figsize=(9, 5))
plt.bar(isimler, farklar)
plt.axhline(0, color="black", linewidth=0.8)
plt.ylabel("mAP50 degisimi (yuzde puan)")
plt.title("Goruntu Iyilestirme Yontemlerinin mAP50 Etkisi")
plt.grid(axis="y", alpha=0.2)
plt.tight_layout()

plt.savefig(
    grafikler / "genel_map50_karsilastirma.png",
    dpi=200
)
plt.close()

# =====================================
# 5. GENEL TABLOYU CSV OLARAK KAYDET
# =====================================

tablo_yolu = sonuclar / "genel_karsilastirma.csv"

with open(tablo_yolu, "w", encoding="utf-8", newline="") as f:
    yazici = csv.writer(f)
    yazici.writerow([
        "Yontem", "Metrik", "Once", "Sonra", "Fark"
    ])

    for deney, sonuc in veriler.items():
        for metrik in metrikler:
            once = sonuc[metrik]["once"]
            sonra = sonuc[metrik]["sonra"]

            yazici.writerow([
                deney,
                metrik,
                round(once, 4),
                round(sonra, 4),
                round(sonra - once, 4)
            ])

print("\nTum grafikler olusturuldu!")
print(f"Grafik klasoru: {grafikler}")
print(f"Genel tablo: {tablo_yolu}")
