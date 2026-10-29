
from ultralytics import YOLO
from insightface.app import FaceAnalysis
import cv2
import os
import numpy as np

# 1. Memuat model deteksi orang
model = YOLO("yolo11n.pt")

# 2. Menyiapkan pengenalan wajah
app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)
app.prepare(ctx_id=0, det_size=(640, 640))

# 3. Membaca semua foto referensi
folder = "referensi"
database = {}

for nama_file in os.listdir(folder):
    if nama_file.lower().endswith((".jpg", ".jpeg", ".png")):
        path = os.path.join(folder, nama_file)
        foto = cv2.imread(path)

        if foto is None:
            print("Foto gagal dibaca:", nama_file)
            continue

        wajah = app.get(foto)

        if len(wajah) == 0:
            print("Wajah tidak ditemukan:", nama_file)
            continue

        nama = os.path.splitext(nama_file)[0]
        database[nama] = wajah[0].normed_embedding

        print("Referensi berhasil dimuat:", nama)

if not database:
    print("Tidak ada foto referensi yang memiliki wajah!")
    raise SystemExit

# 4. Membuka webcam
kamera = cv2.VideoCapture(0)

while kamera.isOpened():
    berhasil, frame = kamera.read()

    if not berhasil:
        print("Kamera tidak dapat dibaca.")
        break

    # Deteksi orang dengan YOLO
    hasil = model.predict(
        source=frame,
        classes=[0],
        verbose=False
    )
    tampilan = hasil[0].plot()

    # Deteksi wajah dengan InsightFace
    wajah_kamera = app.get(frame)

    for wajah in wajah_kamera:
        fitur = wajah.normed_embedding

        nama_terbaik = "Tidak dikenal"
        skor_terbaik = -1

        # Membandingkan dengan semua foto
        for nama, fitur_ref in database.items():
            skor = float(np.dot(fitur_ref, fitur))

            if skor > skor_terbaik:
                skor_terbaik = skor
                nama_terbaik = nama

        # Ambang awal untuk pengujian
        if skor_terbaik < 0.45:
            nama_terbaik = "Tidak dikenal"

        x1, y1, x2, y2 = map(int, wajah.bbox)

        cv2.rectangle(
            tampilan, (x1, y1), (x2, y2),
            (0, 255, 0), 2
        )

        cv2.putText(
            tampilan,
            nama_terbaik,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 255, 0), 2
        )

    cv2.imshow("Monitor APD - Pengenalan Wajah", tampilan)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

kamera.release()
cv2.destroyAllWindows()
