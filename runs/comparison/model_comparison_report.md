# 🥚 Laporan Komparasi: YOLOv8 Nano vs RetinaNet (ResNet-50 FPN)
**Proyek:** Sistem Sortir Mutu Telur Otomatis Berdasarkan SNI 3926:2023  
**Dataset Pengujian:** `D:\github_rama\Egg-SortingSystem\Egg Dataset Clean/test` (90 citra independen)  
**Perangkat:** `cpu` (CPU)  

---

## 1. Tabel Perbandingan Metrik Kinerja

| Parameter Komparasi | YOLOv8 Nano (`yolov8n.pt`) | RetinaNet (`ResNet-50 FPN`) | Selisih / Analisis |
| :--- | :---: | :---: | :---: |
| **Arsitektur Backbone** | Modified CSPDarknet53 | Deep ResNet-50 + FPN | ResNet-50 lebih dalam & berat |
| **Mekanisme Deteksi** | Anchor-Free Decoupled Head | Anchor-Based Multi-scale Head | YOLOv8 lebih modern & fleksibel |
| **Fungsi Loss** | CIoU + DFL + BCE | Focal Loss + Smooth L1 | Focal Loss unggul atasi imbalance |
| **mAP @ 0.50** | **66.84%** | **64.72%** | -2.12% |
| **mAP @ 0.50:0.95** | **39.95%** | **36.76%** | -3.19% |
| **Presisi (Precision)** | 43.13% | 10.52% | Tingkat akurasi deteksi positif |
| **Recall** | 83.19% | 92.05% | Sensitivitas menangkap cacat |
| **F1-Score** | 56.81% | 18.88% | Harmonic mean presisi & recall |
| **AP50 Cracked (Retak)** | 89.51% | 87.85% | Deteksi retakan rambut |
| **AP50 Dirt (Kotoran)** | 44.18% | 41.59% | Deteksi noda kotoran |
| **Ukuran Bobot File** | **6.72 MB** | **123.28 MB** | YOLOv8 ~18.3x lebih ringkas |
| **Jumlah Parameter** | **3.39 Juta** | **32.19 Juta** | RetinaNet memiliki ~9.5x parameter |
| **Latensi Inferensi** | **149.93 ms** | **2596.20 ms** | YOLOv8 ~17.3x lebih cepat |
| **Throughput (FPS)** | **6.7 FPS** | **0.4 FPS** | Kemampuan proses real-time |

---

## 2. Analisis & Pembahasan Akademik (Untuk Laporan Proyek Semester 5)

### A. Karakteristik Arsitektur & Kompleksitas Komputasi
1. **YOLOv8 Nano (`yolov8n`)**:
   - Dirancang khusus untuk efisiensi komputasi tinggi pada edge device (Raspberry Pi, Jetson Nano, atau laptop biasa).
   - Menggunakan pendekatan *anchor-free*, sehingga tidak memerlukan penyesuaian ukuran anchor box buatan dan mampu menggeneralisasi bentuk cacat cangkang telur yang tidak beraturan dengan cepat.
   - Dengan parameter hanya ~3.4M dan ukuran bobot ~6.7 MB, inferensi dapat berjalan di atas konveyor dengan latensi sangat rendah (~149.9 ms).

2. **RetinaNet (`ResNet-50 FPN`)**:
   - Model *one-stage detector* klasik yang menjadi tonggak penting dalam riset computer vision dengan memperkenalkan **Focal Loss** untuk mengatasi masalah ketidakseimbangan kelas (*foreground-background class imbalance*).
   - Memiliki representasi fitur visual yang sangat kaya berkat backbone ResNet-50 dan piramida fitur (FPN P3-P7).
   - Namun, kompleksitas parameternya mencapai ~32.2M (~123.3 MB), menyebabkan waktu inferensi lebih tinggi (~2596.2 ms).

### B. Kesimpulan & Rekomendasi Deployment Sistem Sortir Telur
- **Untuk Implementasi Hardware Nyata (Real-time Conveyor):**  
  **YOLOv8 Nano direkomendasikan sebagai model utama** karena memenuhi syarat kecepatan inspeksi *real-time* konveyor (roller berputar 1–2 putaran per detik) tanpa menyebabkan jeda / bottleneck pada mekanik servo pemilah.
- **Untuk Verifikasi Mutu Resolusi Tinggi (Offline Quality Audit):**  
  **RetinaNet dapat digunakan sebagai model validasi sekunder** di server kontrol pusat untuk menganalisis retakan mikro atau sebagai benchmark pembanding dalam dokumen penelitian ilmiah / skripsi.

---
*Laporan ini dibuat otomatis oleh script `compare_models.py`*
