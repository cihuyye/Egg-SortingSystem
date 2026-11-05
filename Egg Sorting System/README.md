# Sistem Sortir Mutu Telur Otomatis Berbasis AI & SNI 3926:2023

Proyek ini dibuat untuk mengotomatisasi proses penyortiran mutu telur ayam konsumsi di atas konveyor secara objektif, cepat, dan presisi tinggi sesuai standar **SNI 3926:2023**.

Sistem ini menggabungkan **Computer Vision (YOLOv8 + OpenCV)** untuk inspeksi fisik cangkang & geometri, **Decision Engine SNI** untuk vonis mutu otomatis, dan **Large Language Model Google Gemini (`gemini-3.8-flash`)** dengan antarmuka **Chatbot Interaktif** di Web Dashboard untuk berdiskusi langsung dengan Auditor Mutu AI mengenai kualitas telur dan klausul regulasi SNI.

---

## 🏗️ Alur Kerja Sistem (Pipeline)

Proses inspeksi berjalan dalam 4 tahap terintegrasi:

1. **Cek Cangkang Luar 360° (YOLOv8 Nano)**:
   - Mendeteksi kondisi kerabang cangkang dari berbagai sudut pandang rotasi (depan, samping, belakang).
   - Mengklasifikasikan 3 kondisi: `cracked` (retak), `dirt` (kotoran/noda), dan `good` (utuh bersih).
   - Aturan keamanan mutu: jika satu sudut saja ditemukan retak (`cracked`), telur langsung divonis Reject.

2. **Ukur Geometri & Bobot Fisik (OpenCV)**:
   - Menghitung dimensi panjang ($L$) dan lebar ($W$) dalam satuan milimeter.
   - Menghitung *Egg Shape Index* (ESI / SI): $SI = (W / L) \times 100$. Nilai normal SNI adalah **72% – 76%**.
   - Mengestimasi volume ellipsoid dan bobot telur (gram), lalu mengelompokkannya ke kelas bobot SNI: Kecil (< 50g), Sedang (50–60g), Besar (60–65g), Ekstra Besar (> 65g).

3. **Logika Keputusan Mutu SNI 3926:2023**:
   - Menentukan Grade Mutu Akhir:
     - **Mutu I (Grade A)**: Cangkang utuh bersih, Shape Index ideal 72–76%, kantung udara segar < 0.5 cm, bebas bercak darah.
     - **Mutu II (Grade B)**: Cangkang ada sedikit noda kotoran, atau Shape Index di luar batas normal, atau kantung udara 0.5–0.9 cm.
     - **Mutu III (Reject)**: Cangkang retak, atau kantung udara > 0.9 cm, atau terdapat bercak darah/noda asing.
   - Menentukan aksi sinyal aktuator hardware (servo jalur premium, jalur pembersihan, atau jalur reject).

4. **Auditor Cerdas AI & Chatbot Interaktif (Google Gemini `gemini-3.8-flash`)**:
   - Data numerik hasil inspeksi computer vision diserahkan ke Gemini.
   - LLM berperan ketat sebagai **Auditor Mutu & Ahli Standardisasi Telur (SNI 3926:2023)**.
   - Menyajikan narasi laporan audit mutu awal secara otomatis di dashboard.
   - Pengguna / operator dapat berinteraksi tanya-jawab langsung via widget Chatbot (misal: alasan penentuan grade, batas kedalaman kantung udara, keamanan konsumsi, atau arahan aktuator konveyor).

---

## 📂 Susunan Folder Proyek

```text
Egg Sorting System/
├── dashboard/
│   ├── index.html               # Web Dashboard interaktif (Unggah 360°, OpenCV, & Gemini Chatbot)
│   ├── style.css                # Tampilan modern, responsif, & tema cerah egg-warmth
│   └── app.js                   # Client logic, deteksi status API, visualisasi bounding box & Chatbot
├── models/
│   └── best.pt                  # Bobot YOLOv8n hasil fine-tuning (6.5 MB, Precision 100%, Recall 100%)
├── modules/
│   ├── __init__.py
│   ├── shape_size_detector.py   # Modul ukur dimensi & Shape Index (OpenCV)
│   ├── candling_detector.py     # Modul cek isi dalam telur / candling (OpenCV)
│   ├── grading_logic.py         # Logika aturan keputusan mutu SNI 3926:2023
│   └── llm_interpreter.py       # Modul integrasi Gemini LLM & engine Chatbot SNI
├── app_flask.py                 # Server REST API + Web Host berbasis Flask
├── requirements.txt             # Daftar dependensi lengkap (Google Gemini Only)
├── test_api_client.py           # Skrip uji coba client (Status, Predict 360°, & Chatbot)
├── predict_sample.py            # Skrip uji pipeline lokal langsung dari terminal
├── detect_candling_live.py      # Pengujian candling real-time via webcam
├── train_model.py               # Skrip pelatihan model YOLOv8
└── README.md
```

---

## 🚀 Cara Menjalankan

### 1. Instalasi Dependensi
Buka terminal dan install dependensi:
```bash
pip install -r requirements.txt
```

### 2. Mengaktifkan Google Gemini API (`gemini-3.8-flash`)
Setel API key Google Gemini Anda di terminal:

* **Windows Command Prompt (CMD)**:
  ```cmd
  set GEMINI_API_KEY=AIzaSy...
  ```

* **Windows PowerShell**:
  ```powershell
  $env:GEMINI_API_KEY="AIzaSy..."
  ```

*(Catatan: Jika API key belum disetel, sistem tetap berfungsi penuh dengan rule-based offline fallback yang cerdas dan tetap terkunci pada SNI).*

### 3. Jalankan Server Flask & Dashboard
```bash
python app_flask.py
```
Output terminal:
```text
=================================================================
 🚀 EGG QUALITY SORTING & DASHBOARD SERVER AKTIF (FLASK)
 🥚 Standar Regulasi : SNI 3926:2023 (Telur Ayam Konsumsi)
 🤖 LLM Model        : Google Gemini (gemini-3.8-flash)
 🔑 Status API Key   : AKTIF
 👉 Dashboard URL    : http://127.0.0.1:8000
=================================================================
```

Buka peramban (browser) ke **`http://127.0.0.1:8000`**. Dashboard visual akan terbuka.

### 4. Uji Coba Otomatis via Skrip Client
Buka terminal lain dan jalankan:
```bash
python test_api_client.py
```
Skrip ini akan menguji 3 alur:
1. Health check server (`GET /api/v1/status`)
2. Prediksi 360° & penentuan grade SNI (`POST /api/v1/predict`)
3. Percakapan interaktif dengan chatbot Auditor Mutu (`POST /api/v1/chat`)

---

## 🤖 Peran & Desain Prompt LLM (Kebutuhan Presentasi ke Dosen)

* **Model**: Google Gemini `gemini-3.8-flash`.
* **Role**: *Auditor Mutu & Ahli Standardisasi Telur Ayam Konsumsi (SNI 3926:2023)*.
* **Strict Guardrails**:
  1. **Strict Context Locking**: Hanya melayani pertanyaan terkait mutu telur, cacat cangkang, geometri/ESI, candling, keamanan pangan konsumsi, dan SNI 3926:2023. Menolak topik di luar konteks dengan sopan dan tegas.
  2. **Integritas Vonis Grade**: Tidak pernah mengubah vonis Grade Akhir yang dihasilkan oleh sistem aturan/decision engine (menjaga determinisme dan akuntabilitas audit).
  3. **Multi-turn Context Aware**: Memahami konteks telur yang sedang diinspeksi (misal jika ada retakan, ia akan memperingatkan risiko kontaminasi mikroba *Salmonella* dan menyarankan aktuator servo membuang ke wadah reject).
