# Sistem Sortir Mutu Telur Otomatis

Proyek ini dibuat untuk menggantikan proses sortir telur manual yang lambat dan subjektif. Kamera memotret cangkang telur di atas roller conveyor secara multi-sudut (360°), kemudian program langsung menilai kelayakan dan tingkat mutu telur berdasarkan standar **SNI 3926:2023** (Telur Ayam Konsumsi).

---

## 🛠️ Cara Kerja Sistem

Sistem menggunakan kamera industri yang dipasang di atas konveyor dengan pencahayaan terkontrol (*inspection chamber*). Setiap telur yang lewat dianalisis melalui 2 tahap inspeksi *computer vision*:

1. **Segmentasi Cacat Cangkang Luar (YOLOv8-seg)**:
   Model deep learning *instance segmentation* (`yolov8n-seg.pt`) mendeteksi dan mensegmentasi area poligon cacat permukaan cangkang telur secara presisi:
   - `cracked` = Retak kerabang (termasuk retakan garis rambut tipis)
   - `dirt` = Noda kotoran / feses ayam (disegmentasi batas poligon)
   - `good` = Cangkang mulus, utuh, bersih tanpa cacat (*negative background sample*)

2. **Pengukuran Dimensi, Bentuk & Bobot (OpenCV Geometry)**:
   OpenCV mengekstraksi kontur cangkang telur dan menghitung:
   - Panjang (*Major Axis*) dan Lebar (*Minor Axis*) dalam milimeter
   - **Egg Shape Index (ESI)**: Rasio lebar dibagi panjang dikali 100%. Menurut SNI 3926:2023, bentuk ideal berkisar antara 72% sampai 76%.
   - **Estimasi Bobot Telur**: Dikelompokkan ke Tabel 2 SNI 3926:2023 (Kecil <50g, Sedang 50-55g, Besar 56-60g, Ekstra Besar >60g).

---

## 📋 Klasifikasi Tingkat Mutu (SNI 3926:2023)

Berdasarkan Tabel 1 dan Tabel 2 SNI 3926:2023:
* **Mutu I (Grade A)**: Cangkang utuh, bersih mulus bebas noda kotoran, dan Shape Index ideal (72–76%).
* **Mutu II (Grade B)**: Cangkang utuh, boleh sedikit bernoda kotoran ringan (toleransi ≤ 1/32 luas permukaan) atau bentuk sedikit di luar batas normal.
* **Mutu III (Reject)**: Cangkang retak (*cracked*) atau berlumur kotoran parah (> 1/16 luas permukaan).

---

## 📂 Susunan Folder Proyek

```text
Egg-SortingSystem/
├── dashboard/
│   ├── index.html               # Web Dashboard cerah (Inspeksi 360° & Grading SNI)
│   └── assets/                  # Komponen frontend telemetry
├── models/
│   └── best.pt                  # Bobot YOLOv8-seg hasil training (Instance Segmentation)
├── modules/
│   ├── __init__.py
│   ├── shape_size_detector.py   # Modul ukur dimensi, ESI & fraktur (OpenCV)
│   ├── grading_logic.py         # Logika penilaian mutu akhir SNI 3926:2023
│   └── llm_interpreter.py       # Asisten Auditor Mutu (Google Gemini)
├── app_flask.py                 # Server REST API + Web Dashboard Host (Flask)
├── predict_sample.py            # Script uji pipeline mandiri dari citra sampel
├── Train_YOLOv8_Segmentation_Colab.ipynb # Notebook pelatihan Google Colab
└── README.md
```

---

## 🚀 Cara Menjalankan

### 1. Jalankan Web Dashboard & Server REST API

```bash
python app_flask.py
```
Buka browser ke **`http://127.0.0.1:8000`** untuk membuka Dashboard Telemetri Produksi & Klasifikasi Real-Time.

### 2. Tes Pipeline Mandiri (Foto Sampel)

```bash
python predict_sample.py
```
Program akan memuat model YOLOv8-seg, mengukur geometri cangkang telur, dan menampilkan hasil grade akhir SNI beserta rekomendasi aksi aktuator konveyor.

### 3. Dokumentasi REST API

Endpoint API dapat diakses di:
- `GET  /api/v1/status` : Cek kesehatan server & status model
- `GET  /api/v1/model-info` : Informasi detail arsitektur & kelas model
- `POST /api/v1/predict` : Kirim citra multi-sudut untuk inspeksi 360°
- `POST /api/v1/chat` : Tanya-jawab interaktif dengan Auditor Mutu (Google Gemini)
