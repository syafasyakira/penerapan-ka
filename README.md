# TrafficVision — Monitoring Kepadatan Lalu Lintas (React + YOLO11)

Web aplikasi end-to-end untuk **mendeteksi jumlah kendaraan yang terlihat dalam frame video**. Dibuat dengan React + Vite (frontend), FastAPI + Ultralytics YOLO11m (backend), dan SQLite (riwayat).

## Fitur

- Dashboard statistik real data dan grafik.
- Unggah video lokal, kamera webcam browser, atau gambar.
- Inferensi YOLO11m di backend; bounding boxes, confidence, filter kelas Mobil / Motor / Bus / Truk.
- ROI poligon dan deduplikasi bounding box sesuai pendekatan notebook.
- Normalisasi frame 1333 × 750 menggunakan scaling dan padding kanan/bawah jika diperlukan.
- Status Sepi / Sedang / Ramai berbasis Q1 dan Q3; **tidak mengarang nilai ambang dataset**.
- Impor `thresholds.json` dari notebook atau kalkulasi kuartil dari riwayat.
- Penyimpanan riwayat SQLite, grafik per frame, agregasi berdasarkan jam, pencarian dan ekspor CSV.
- Pengaturan model, confidence, ROI, resolusi YOLO, luas bounding box.

## Instalasi

Persyaratan: **Node.js 18+**, **Python 3.10+**, dan akses internet saat mengunduh library + bobot model YOLO pertama kali.

### 1. Backend (Terminal 1)

```bash
cd backend
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

API / dokumentasi: http://127.0.0.1:8000/docs

### 2. Frontend (Terminal 2)

```bash
cd frontend
npm install
npm run dev
```

Buka **http://localhost:5173**. Server Vite mem-proxy `/api` ke backend `127.0.0.1:8000`.

### 3. Coba monitoring

1. Buka menu **Live Monitoring**.
2. Unggah video `.mp4` / `.webm`, atau aktifkan webcam.
3. Klik **Mulai Analisis**. Frame dikirim berkala ke backend.
4. Lihat bounding box, hitungan per jenis kendaraan, dan grafik/dashboard.
5. Di **Pengaturan Model**, impor file `thresholds.json` hasil notebook agar status kepadatan dapat muncul, atau hitung Q1/Q3 dari sekurang-kurangnya 4 observasi.
6. Buka **Analitik & Riwayat** untuk ekspor hasil `.csv`.

## Kesamaan dengan notebook dan batasan

- Default `yolo11m.pt`, kelas COCO `[2,3,5,7]`, `conf=0.10`, `imgsz=1280`, `MIN_AREA=600`, `IOU=0.4`, `OVERLAP=0.6`.
- ROI default: `(0,.5), (.6,.2), (.65,.2), (.6,1), (0,1)`. Filter memakai titik **tengah-bawah** bounding box.
- Ukuran normalisasi 1333 × 750. Pada notebook normalisasi dengan *right padding*, implementasi backend memakai kanvas hitam untuk padding kanan/bawah. Jika dataset sangat berbeda, sesuaikan ROI.
- **Hasil adalah kendaraan yang tampak pada sebuah frame**, bukan kendaraan unik yang melintas per menit dan bukan satuan kendaraan/km. Jika kendaraan yang sama muncul di 10 frame, riwayat menghitungnya 10 kali. Tracking + garis hitung perlu ditambahkan untuk mengukur arus.
- **Kepadatan relatif** berdasarkan distribusi counts dataset; status belum tersedia hingga Q1 dan Q3 dimasukkan. Ini mengikuti interpretasi notebook.
- Puncak jam dapat diteliti menggunakan agregasi rata-rata per jam; UI menyediakan grafiknya. Timestamp disimpan dalam UTC; grafik agregasi per jam ditampilkan dalam WIB (Asia/Jakarta).
- Kecepatan tergantung GPU/CPU. YOLO11m pada CPU, imgsz 1280, dapat membutuhkan waktu lebih lama dibanding interval frame yang dipilih. Request berikutnya dilewati saat inference masih berjalan.
- Untuk webcam, gunakan `localhost` atau HTTPS. Video diproses lokal dalam browser; **sampel frame** dikirim ke server. Webcam memerlukan izin pengguna.
- Tidak ada backend video stream RTSP langsung; gunakan webcam browser atau unggah video. Untuk RTSP/CCTV produksi diperlukan konektor stream tambahan dan kontrol autentikasi.
- Jika hosting publik, tambahkan autentikasi, HTTPS, batas akses endpoint, kebijakan privasi kamera, dan manajemen retensi data.

## Struktur folder

```text
frontend/                  React + Vite
  src/App.jsx              Halaman dashboard, monitoring, analytics, settings
  src/styles.css           Dark UI responsive
backend/
  main.py                  FastAPI, YOLO inference, SQLite
  requirements.txt
  config.json              Dibuat otomatis setelah save settings
  traffic.db               Dibuat otomatis setelah server pertama jalan
```

## Troubleshooting

- **API Tidak Terhubung**: pastikan backend berjalan di port 8000.
- **Ultralytics belum terinstal**: aktifkan venv, jalankan `pip install -r requirements.txt`.
- **Model lambat**: pilih `yolo11n.pt` atau `imgsz=640` di pengaturan (hasil bisa berbeda dari notebook).
- **Bobot model**: Ultralytics akan mengunduh `yolo11m.pt` saat inferensi pertama kali apabila bobot belum tersedia.
- **Webcam gagal**: cek izin kamera pada browser dan pastikan memakai localhost atau HTTPS.
- **Klasifikasi belum muncul**: impor `thresholds.json` dan klik Simpan.
