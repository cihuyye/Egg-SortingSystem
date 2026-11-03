# Model Pembanding: RetinaNet (ResNet-50 FPN V2)

Folder ini khusus didedikasikan untuk eksperimen pembanding (*benchmark*) melawan model utama **YOLOv8 Nano** pada sistem penyortir mutu telur berbasis **SNI 3926:2023**.

---

### Tujuan Pembanding

Membandingkan performa dua arsitektur deteksi objek satu tahap (*single-stage detector*):
1. **YOLOv8 Nano (Model Utama)**: Arsitektur modern *anchor-free*, decoupled head, dioptimalkan untuk perangkat *edge* dan konveyor real-time berkecepatan tinggi.
2. **RetinaNet ResNet-50 FPN V2 (Model Pembanding)**: Arsitektur seminal *anchor-based* dengan *Focal Loss* (Lin et al., 2017) untuk menangani ketimpangan latar belakang.

---

### Struktur Folder Ini

```text
RetinaNet/
├── convert_yolo_to_coco.py      # Skrip konversi dataset YOLO TXT ke MS-COCO JSON
├── train_retinanet_colab.py     # Skrip pelatihan RetinaNet PyTorch di Google Colab
├── train_retinanet_colab.ipynb  # Notebook Colab siap pakai (upload & run)
└── README.md                    # Dokumentasi ini
```

---

### Panduan Penggunaan

#### 1. Konversi Dataset (Jika Ingin Dijalankan Ulang)
Jalankan skrip konversi di terminal:
```bash
python convert_yolo_to_coco.py
```
Dataset berformat COCO akan dibuat di folder `Egg Dataset COCO/` dengan 3 subset:
* `annotations/instances_train.json` (562 citra)
* `annotations/instances_val.json` (160 citra)
* `annotations/instances_test.json` (80 citra)

#### 2. Pelatihan di Google Colab
1. Buka [Google Colab](https://colab.research.google.com).
2. Upload file **`train_retinanet_colab.ipynb`** (atau copy isi `train_retinanet_colab.py`).
3. Set Runtime ke **T4 GPU** (*Runtime -> Change runtime type -> T4 GPU*).
4. Upload file **`Egg_Dataset_COCO.zip`** (tersedia di root proyek).
5. Jalankan sel pelatihan (50 epoch).

---

### Target Metrik Perbandingan untuk Laporan/Sidang

| Parameter Evaluasi | YOLOv8 Nano | RetinaNet (ResNet-50 FPN) | Analisis Akademik |
| :--- | :---: | :---: | :--- |
| **Precision** | **99.97%** | *Diuji di Colab* | Membandingkan tingkat *false alarm* |
| **Recall** | **100.0%** | *Diuji di Colab* | Kemampuan menangkap retakan cangkang |
| **Latensi Inferensi** | **~5.6 ms** | *Diuji di Colab* | Kesesuaian dengan konveyor fisik |
| **Ukuran Bobot** | **6.5 MB** | **~135 MB** | Efisiensi penyimpanan memori |
| **Jumlah Parameter** | **~3.2 Juta** | **~31.5 Juta** | Kompleksitas komputasi model |

