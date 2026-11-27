# 🥚 Sistem Klasifikasi & Sortir Mutu Telur Otomatis (SNI 3926:2023)
> **Integrasi 3 Mata Kuliah Workshop:**  
> 1. **Workshop Pengolahan Citra dan Vision**  
> 2. **Workshop Sistem Cerdas**  
> 3. **Workshop Sistem Tertanam**  

Proyek ini adalah sistem inspeksi dan sortir mutu telur ayam konsumsi otomatis berbasis standar **SNI 3926:2023 (Telur Ayam Konsumsi - Mutu dan Penanganan)**. Sistem mengintegrasikan kamera inspeksi multi-sudut (360°), pemrosesan citra digital OpenCV, model deep learning YOLOv8-seg, sistem pendukung keputusan (*Decision Engine*), mikrokontroler aktuator pemilah fisik, serta asisten auditor interaktif berbasis **Google Gemini LLM**.

---

## 🏛️ Arsitektur & Integrasi 3 Workshop

```
[ TELUR DI KONVEYOR ]
         │
         ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 1. WORKSHOP SISTEM TERTANAM (HARDWARE & AKUISISI)                      │
│ - Modul Kamera ESP32-CAM (Akuisisi video stream multi-sudut 360°)      │
│ - Konveyor Roller Pemutar Otomatis                                     │
│ - Aktuator Pemilah: Gate 1 (Mutu I), Gate 2 (Mutu II), Ejector (Afkir) │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │ Frame Citra (RGB)
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 2. WORKSHOP PENGOLAHAN CITRA & VISION (OPENCV PIPELINE)                │
│ - Preprocessing: Ruang Warna CIE L*a*b* (CLAHE), HSV, Gaussian Blur    │
│ - Segmentasi: Adaptive Otsu Binarization + Morphological Closing/Open  │
│ - Filter Retak Mikro: Directional Vertical Blackhat (Kernel 3x21)      │
│ - Ekstraksi Fitur:                                                     │
│   • Geometri: Major Axis (L), Minor Axis (W), Area, Perimeter          │
│   • Bentuk: Egg Shape Index (ESI % = W/L * 100), Convex Hull, Solidity │
│   • Fisika/Massa: Volume Elipsoid (pi/6 * L * W²) & Estimasi Bobot     │
│   • Invarian: Hu Moments (H1-H7 invarian terhadap rotasi & skala)      │
│   • Warna: Mean HSV (Validasi cangkang cokelat vs putih gading)        │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │ Mask, Dimensi, ESI, Bobot
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 3. WORKSHOP SISTEM CERDAS (AI & DECISION ENGINE)                       │
│ - Deep Learning: YOLOv8-seg (Deteksi cacat retak 'cracked' & 'dirt')   │
│ - Multi-Angle Consensus Engine: Fusi 360° (Worst-Case Defect Priority) │
│ - Decision Engine SNI 3926:2023: Vonis Mutu I / Mutu II / Mutu III     │
│ - Generative AI: Google Gemini LLM (Auditor Mutu AI dengan 7 pilar     │
│   Anti-Halusinasi & SDK resmi google-genai)                            │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │ Sinyal Klasifikasi & Aksi
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 4. AKTUASI & ANTARMUKA PENGGUNA                                        │
│ - Sinyal Aktuator: Pulsa Kontrol Servo / Solenoid Pneumatik Mekanik    │
│ - Frontend UI: EggSort-AI Industrial Dashboard (Vue 3 + Vite)          │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 📋 Standar Mutu Telur (SNI 3926:2023)

Sistem merujuk secara ketat pada klausul **SNI 3926:2023**:

### Tabel 1: Syarat Mutu Kerabang / Cangkang
| Faktor Mutu | Mutu I (Grade A) | Mutu II (Grade B) | Mutu III (Reject / Afkir) |
|---|---|---|---|
| **Keutuhan Kerabang** | Utuh, tidak retak | Utuh, tidak retak | **Retak / Pecah** |
| **Kebersihan Kerabang** | Bersih total, bebas noda | Sedikit noda kotoran ($\le 1/32$ luas permukaan) | Noda kotoran parah ($> 1/16$ luas permukaan) |
| **Indeks Bentuk (ESI)** | Normal ($72\% - 76\%$) | Agak lonjong ($<72\%$) / bulat ($>76\%$) | Cacat bentuk ekstrem |
| **Peruntukan Pasar** | Konsumsi Meja Segar / Ritel Berlabel | Industri Olahan (Bakery, Mie, Tepung Telur) | Karantina Pangan / Afkir (Dilarang Konsumsi Segar) |
| **Aksi Aktuator** | **Gate 1** (Wadah Ritel) | **Gate 2** (Wadah Industri) | **Reject Ejector** (Wadah Karantina) |

### Tabel 2: Klasifikasi Bobot Telur
* **Sangat Besar (Jumbo)**: $> 60\text{ gram}$
* **Besar**: $55 - 60\text{ gram}$
* **Sedang**: $50 - 54.9\text{ gram}$
* **Kecil**: $< 50\text{ gram}$

---

## 📂 Struktur Direktori Proyek

```text
Egg-SortingSystem/
├── app_flask.py                 # Backend REST API Server (Port 8000)
├── compare_models.py            # Skrip komparasi head-to-head YOLOv8 vs RetinaNet
├── demo_ekstraksi_fitur.py      # Skrip demo & pembuktian visual 6 panel ekstraksi fitur
├── uji_pembuktian_telur.py      # Skrip pengujian pembuktian fusi multi-sudut (Sudut 1 & 2)
├── START_SYSTEM.bat             # Otomasi startup sistem (Flask API + Vue Dashboard)
├── run_flask.bat                # Script peluncuran Flask API backend
├── requirements.txt             # Dependensi Python resmi
├── .env                         # Konfigurasi GEMINI_API_KEY
│
├── yolov8/                      # 🌟 ARTEFAK & PELATIHAN YOLOV8
│   ├── Train_YOLOv8_Segmentation_Colab.ipynb  # Notebook Colab pelatihan YOLOv8-seg
│   ├── weights/
│   │   └── best.pt              # Bobot model YOLOv8-seg (~6.7 MB)
│   ├── matriks/                 # Grafik evaluasi (BoxF1, MaskPR, Confusion Matrix, dll.)
│   ├── __init__.py
│   └── README.md
│
├── retinanet/                   # 🌟 ARTEFAK & PELATIHAN RETINANET (KOMPARASI AKADEMIK)
│   ├── Train_Egg_Defect_RetinaNet_Colab.ipynb  # Notebook Colab pelatihan RetinaNet GPU
│   ├── train_retinanet.py       # Script pelatihan RetinaNet lokal (CUDA GPU)
│   ├── train_retinanet_cpu.py   # Script pelatihan RetinaNet lokal fallback CPU
│   ├── eval_retinanet.py        # Script evaluasi independen test set
│   ├── predict_retinanet.py     # Script inferensi deteksi objek RetinaNet
│   ├── retinanet_dataset.py     # PyTorch Dataset loader (konversi format YOLO ke BBox)
│   ├── retinanet_metrics.py     # Perhitungan mAP50, mAP50-95, Precision, Recall
│   ├── retinanet_model.py       # Arsitektur ResNet-50 FPN Torchvision Detection
│   ├── weights/                 # Checkpoint bobot best.pt
│   ├── matriks/                 # Grafik evaluasi (results.png, confusion_matrix, dll.)
│   ├── __init__.py
│   └── README.md
│
├── modules/                     # Modul internal backend
│   ├── __init__.py
│   ├── shape_size_detector.py   # Pipeline Pengolahan Citra, Morfologi, ESI, & Fisika Bobot
│   ├── grading_logic.py         # Decision Engine Klasifikasi Akhir Mutu SNI 3926:2023
│   └── llm_interpreter.py       # Modul Asisten Auditor Mutu (Gemini v3.5+ & Anti-Halusinasi)
├── models/
│   └── best.pt                  # Bobot runtime aktif YOLOv8-seg untuk server Flask
├── Egg Dataset Clean/           # Dataset bersih bebas data leakage (Train, Valid, Test)
└── Egg_Dataset_Clean.zip        # Arsip dataset siap upload ke Google Colab (23.55 MB)
```

---

## 🧪 Pembuktian Ekstraksi Fitur (Pengolahan Citra & Vision)

Tahap ekstraksi fitur dibuktikan secara kuantitatif melalui skrip otomatis:

```powershell
python demo_ekstraksi_fitur.py
```

Skrip ini mengekstraksi **13 vektor fitur fisik** dari cangkang telur:
1. **Luas Kontur (Area)** via `cv2.contourArea` ($px$)
2. **Keliling (Perimeter)** via `cv2.arcLength` ($px$)
3. **Convex Hull & Solidity**: $\text{Solidity} = \frac{\text{Area Kontur}}{\text{Convex Hull Area}}$ (Validasi kecembungan cangkang $\ge 0.84$)
4. **Major Axis ($L$) & Minor Axis ($W$)**: Panjang dan lebar milimeter via `cv2.fitEllipse`
5. **Egg Shape Index ($ESI$)**: $(W / L) \times 100\%$ (SNI Lampiran A.1.1.3.1)
6. **Estimasi Volume Elipsoid**: $V = \frac{\pi}{6} \cdot L \cdot W^2$ ($\text{cm}^3$)
7. **Estimasi Bobot**: $V \times 1.035\text{ g/cm}^3$ (Tabel 2 SNI)
8. **Nilai Rata-rata HSV ($H, S, V$)**: Validasi kromatik warna cangkang cokelat vs putih
9. **Keseragaman Tekstur**: Standar deviasi intensitas RGB

> Hasil visualisasi 6 panel disimpan ke file gambar: **`bukti_ekstraksi_fitur.png`**.

---

## 🔬 Pengujian Fusi Multi-Sudut (360° Inspection)

Untuk membuktikan keandalan sistem terhadap variasi orientasi dan latar belakang:

```powershell
python uji_pembuktian_telur.py
```

### Hasil Pengujian Nyata 1 Telur dengan 2 Sudut:
* **Sudut 1 (Latar Keramik Berbayang, Miring $116^\circ$)**: Terdeteksi bintik noda cangkang $\rightarrow$ `DIRT` ($98.5\%$).
* **Sudut 2 (Latar Kertas Putih, Rebah $88^\circ$)**: Terdeteksi retakan fisik cangkang $\rightarrow$ `CRACKED` ($98.5\%$).
* **Invarian Bentuk**: $ESI_1 = 80.88\%$ vs $ESI_2 = 82.52\%$ (Selisih hanya $1.64\%$), nilai **Hu Moment 1** selisih hanya $0.0016$ membuktikan bentuk geometris teridentifikasi identik.
* **Fusi Multi-Sudut (Decision Engine)**: Telur divonis **MUTU III (REJECT)** karena retakan kerabang mendominasi faktor kotoran (prioritas bahaya *Salmonella sp.*).

> Gambar visualisasi komparasi disimpan ke: **`uji_pembuktian_telur.png`**.

---

## 🤖 Modul Auditor Mutu AI (Google Gemini LLM)

Terintegrasi pada file `modules/llm_interpreter.py` menggunakan **SDK resmi `google-genai` (v3.6.1)**:
* **Model Utama**: `gemini-3.5-flash`
* **Auto-Fallback Chain**: `gemini-3.5-flash` $\rightarrow$ `gemini-2.5-flash` $\rightarrow$ `gemini-flash-latest` (otomatis beralih jika limit kuota tercapai).
* **7 Lapisan Guardrail Anti-Halusinasi**:
  1. Suhu deterministik (`temperature = 0.2`, `top_p = 0.8`).
  2. Grounding ketat pada data numerik sensor komputer vision.
  3. Filter kata kunci negatif (menolak topik di luar telur: coding, politik, dll).
  4. Pembatas riwayat chat maksimal 10 turn.
  5. Validasi konsistensi vonis grade (AI dilarang mengubah keputusan Decision Engine).
  6. Filter keselamatan pangan (*Safety Post-Filter* bahaya bakteri patogen).
  7. Engine offline cerdas (tetap menjawab detail SNI saat koneksi terputus).

---

## 🚀 Panduan Menjalankan Sistem

### 1. Prasyarat & Instalasi Dependensi
Pastikan Python 3.10+ terpasang, lalu instal dependensi:
```powershell
pip install -r requirements.txt
```

Atur API Key Google Gemini di file `.env`:
```env
GEMINI_API_KEY=AIzaSy...
```

### 2. Jalankan Backend Flask REST API
```powershell
python app_flask.py
```
Server aktif pada **`http://127.0.0.1:8000`**.

### 3. Dokumentasi REST API Endpoints
* **`GET /api/v1/status`**: Cek status server, ketersediaan model YOLOv8, dan koneksi LLM.
* **`GET /api/v1/model-info`**: Informasi detail arsitektur YOLOv8, metrik akurasi mAP50, dan OpenCV.
* **`POST /api/v1/predict`**: Mengirim 1 atau beberapa file citra telur (inspeksi multi-sudut 360°). Mengembalikan vonis mutu SNI, ESI, bobot, bounding box cacat, dan narasi interpretasi LLM.
* **`POST /api/v1/chat`**: Endpoint interaksi tanya-jawab interaktif dengan Auditor Mutu AI.

### 4. Menjalankan Frontend Dashboard
Buka direktori frontend terpisah (`eggsort-dashboard`):
```powershell
cd ../eggsort-dashboard
npm install
npm run dev
```
Buka peramban ke **`http://localhost:5173/`**.
