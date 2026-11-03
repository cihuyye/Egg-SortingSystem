# Sistem Sortir Mutu Telur Otomatis

Proyek kelompok 3 ini dibuat untuk menggantikan proses sortir telur manual yang lambat dan tidak konsisten. Idenya sederhana: kamera memotret telur di atas roller conveyor, terus program langsung menilai layak jual atau tidak berdasarkan aturan SNI 3926:2023.

## Gimana Cara Kerjanya

Sistemnya pakai 1 kamera samping yang dipasang di dalam kotak gelap (dark box) di atas konveyor. Tiap telur yang lewat difoto dua kali: sekali pakai lampu terang buat lihat cangkang luar, sekali pakai lampu bawah (candling) buat lihat isi dalam.

Prosesnya ada 3 tahap:

1. **Cek Cangkang Luar (YOLOv8 Nano)** — Model deep learning ini yang ngecek permukaan cangkang telur. Dia bisa bedain 3 kondisi:
   - `cracked` = retak (termasuk retakan rambut yang tipis banget)
   - `dirt` = kotor / ada noda kotoran ayam
   - `good` = mulus, bersih, ga ada cacat

2. **Ukur Dimensi & Bentuk (OpenCV)** — Bagian ini ngitung panjang, lebar, Shape Index (SI), sama perkiraan berat telur. SI itu rasio lebar dibagi panjang dikali 100. Menurut SNI 3926:2023, telur normal punya SI antara 72 sampai 76. Kalau di luar itu berarti bentuknya kelewat lonjong atau kelewat bulat.

3. **Candling / Peneropongan (OpenCV)** — Ini yang ngecek bagian dalam telur pakai sinar lampu dari bawah. Yang dicek:
   - Kedalaman kantung udara (makin dalam = makin lama disimpan = makin ga segar)
   - Ada tidaknya bercak darah atau noda asing di dalam

Setelah ketiga tahap selesai, program langsung menentukan grade akhir sesuai SNI 3926:2023:
- **Mutu I (Grade A)** — Cangkang utuh bersih, SI normal 72-76, kantung udara kurang dari 0,5 cm, dan bebas bercak darah.
- **Mutu II (Grade B)** — Cangkang boleh agak kotor, SI boleh sedikit di luar batas, kantung udara 0,5-0,9 cm.
- **Mutu III (Reject)** — Cangkang retak, kantung udara lebih dari 0,9 cm, atau ada bercak darah di dalam.

## Susunan Folder

```text
Egg Sorting System/
├── dashboard/
│   ├── index.html               # Web Dashboard cerah (Inspeksi 360° & Grading SNI)
│   └── README.md
├── models/
│   └── best.pt                  # Bobot YOLOv8 hasil training di Colab (6.5 MB)
├── modules/
│   ├── __init__.py
│   ├── shape_size_detector.py   # Modul ukur dimensi & Shape Index (OpenCV)
│   ├── candling_detector.py     # Modul cek isi dalam telur / candling (OpenCV)
│   └── grading_logic.py         # Logika penilaian mutu akhir SNI 3926:2023
├── app_api.py                   # Server REST API + Web Dashboard Host (FastAPI + Uvicorn)
├── predict_sample.py            # Script tes pipeline lengkap dari foto sampel
├── detect_candling_live.py      # Dashboard candling real-time pakai webcam
├── train_model.py               # Script training model YOLOv8
├── test_api_client.py           # Script tes kirim gambar ke API
└── README.md
```

## Hasil Training Model YOLOv8

Training dilakukan di Google Colab pakai GPU Tesla T4. Settingannya:
- Dataset: 802 foto telur (562 train, 160 validasi, 80 tes)
- Resolusi input: 640x640 piksel
- Batch size: 16
- Jumlah epoch: 50

Hasilnya cukup memuaskan:
- Precision: 100% — ga ada satupun deteksi palsu
- Recall: 100% — semua telur retak berhasil terdeteksi
- mAP50: 99.5%
- mAP50-95: 99.5%
- Kecepatan inferensi: sekitar 5.6 ms per gambar di GPU, di CPU laptop juga masih lancar

## Cara Menjalankan

### Tes Pipeline Lengkap (dari foto sampel)

```bash
cd "D:\Egg Sorting System"
python predict_sample.py
```

Nanti program bakal jalanin deteksi cangkang, ukur dimensi, analisis candling, terus tampilkan grade akhirnya (Mutu I / II / III) beserta alasan dan aksi servo.

### Jalankan Web Dashboard & API (Untuk Demo)

Cara paling praktis untuk demo:
```bash
python app_api.py
```
Lalu tekan tombol **`Ctrl` dan klik link `http://127.0.0.1:8000`** yang muncul di terminal. Web dashboard cerah akan langsung terbuka di browser.

### Tes REST API Endpoint (Swagger Docs)

Buka browser ke `http://127.0.0.1:8000/docs` buat coba kirim gambar lewat Swagger UI. Atau pakai Postman, kirim POST ke `/api/v1/predict` dengan form-data key `file` atau `files` (bisa beberapa foto untuk inspeksi 360°).

### Dashboard Candling Live (Webcam)

```bash
python detect_candling_live.py
```

Tekan `q` di jendela video buat keluar.

## Catatan & PR yang Masih Dikerjakan

1. **Kalibrasi Konversi Piksel ke Milimeter** — Nilai `px_to_mm` sekarang masih pakai angka estimasi (0.15). Nanti harus dikalibrasi ulang sesuai jarak kamera ke roller konveyor yang sebenarnya.
2. **Posisi Candling** — Pas candling, telur harus berhenti sekitar 0.3 detik tepat di atas lubang lampu bawah supaya ga silau dan ga goyang. Rencananya konveyor bakal dihentikan sesaat lewat perintah dari Arduino.
3. **Threshold Candling** — Batas gelap-terang buat deteksi kantung udara dan bercak darah perlu dicoba langsung pakai telur asli dengan tingkat kesegaran yang beda-beda.
4. **Integrasi Hardware** — Belum disambungin ke Arduino/ESP32 buat kontrol servo pemilah 3 jalur (Mutu I, Mutu II, Reject).
