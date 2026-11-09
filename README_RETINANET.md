# 🥚 Panduan Pelatihan & Komparasi Model RetinaNet vs YOLOv8 Nano
**Proyek:** Sistem Sortir Mutu Telur Otomatis Berdasarkan SNI 3926:2023  
**Semester 5 Project**

Repository ini telah dilengkapi dengan pipeline lengkap untuk melatih, mengevaluasi, dan membandingkan model **RetinaNet (ResNet-50 FPN)** secara *head-to-head* dengan model **YOLOv8 Nano** yang sudah Anda latih sebelumnya.

---

## 📁 Struktur Berkas Terkait RetinaNet & Komparasi

```text
Egg Sorting System/
├── Egg Dataset Clean/                    # Dataset bersih (717 train, 87 valid, 88 test)
│   ├── train/ (images & labels)
│   ├── valid/ (images & labels)
│   ├── test/  (images & labels)
│   └── annotations/                     # Format standar COCO JSON (hasil konversi)
│       ├── instances_train.json
│       ├── instances_valid.json
│       └── instances_test.json
├── retinanet_dataset.py                 # PyTorch Dataset loader (baca poligon YOLO & neg sample)
├── retinanet_model.py                   # Arsitektur RetinaNet ResNet-50 FPN Torchvision
├── retinanet_metrics.py                 # Evaluator metrik mAP50, mAP50-95, Precision, Recall
├── train_retinanet.py                   # Script pelatihan lokal model RetinaNet
├── eval_retinanet.py                    # Script evaluasi independen pada test set
├── predict_retinanet.py                 # Script inferensi visual & class detector
├── compare_models.py                    # Script komparasi head-to-head YOLOv8 vs RetinaNet
├── yolo_to_coco.py                      # Konverter dataset YOLOv8 ke COCO JSON
├── Train_Egg_Defect_RetinaNet_Colab.ipynb # Notebook Google Colab (Train + Visual Compare)
└── Egg Sorting System/
    ├── models/
    │   ├── best.pt                      # Bobot model YOLOv8 Anda (6.5 MB)
    │   └── best_retinanet.pth           # Bobot model RetinaNet Anda
    └── modules/
        └── retinanet_detector.py        # Adapter RetinaNet untuk sistem sortir Flask/OpenCV
```

---

## 🔄 Bagaimana Dataset YOLOv8 Diadaptasi untuk RetinaNet

Dataset `Egg Dataset Clean` awalnya dibuat untuk YOLOv8 dengan karakteristik:
1. **Format Poligon Segmen**: Berisi deretan titik `class_id x1 y1 x2 y2 ...` hasil ekspor LabelMe (`json_to_yolo.py`).
2. **Negative Samples**: Folder `good` berisi file `.txt` kosong berukuran 0-byte (telur mulus tanpa bounding box).

File [`retinanet_dataset.py`](file:///e:/Semester%205/Project/Egg%20Sorting%20System/retinanet_dataset.py) secara otomatis:
- Menghitung bounding box pembungkus minimal `[xmin, ymin, xmax, ymax]` dari poligon:
  $$\text{xmin} = \min(x) \times W, \quad \text{xmax} = \max(x) \times W$$
  $$\text{ymin} = \min(y) \times H, \quad \text{ymax} = \max(y) \times H$$
- Memetakan ID kelas sesuai standar Torchvision Detection:
  - `0`: Background (negative sample telur normal)
  - `1`: `cracked` (retak)
  - `2`: `dirt` (kotoran)
- Menangani gambar tanpa cacat (`good`) secara aman (target berupa tensor kosong `[0, 4]`).
- Melakukan resize ke 640×640 piksel dan augmentasi data (horizontal flip, brightness/contrast jitter) secara sinkron antara citra dan bounding box.

---

## 🚀 Cara Menjalankan Pelatihan RetinaNet

### Opsi 1: Latih di Google Colab (Sangat Direkomendasikan jika Menggunakan GPU T4)
1. Buka [Google Colab](https://colab.research.google.com/).
2. Unggah file notebook [`Train_Egg_Defect_RetinaNet_Colab.ipynb`](file:///e:/Semester%205/Project/Egg%20Sorting%20System/Train_Egg_Defect_RetinaNet_Colab.ipynb).
3. Unggah file zip dataset `Egg_Dataset_Clean.zip` ke panel Files Colab.
4. Jalankan seluruh cell (*Runtime > Run all*):
   - Cell 1-2: Cek GPU & ekstrak dataset.
   - Cell 3-5: Training 50 epoch dengan AdamW & Cosine Annealing LR.
   - Cell 6-7: Visualisasi kurva loss dan deteksi pada test set.
   - Cell 8: Unduh otomatis `best_retinanet.pth`.
   - Cell 9: Komparasi langsung dengan mengunggah `best.pt` YOLOv8 Anda.

### Opsi 2: Latih di Komputer Lokal (CLI)
Jika komputer Anda memiliki GPU NVIDIA (CUDA):

```bash
cd "e:\Semester 5\Project\Egg Sorting System"

# Pelatihan standar (50 epoch, batch size 8, resolusi 640)
python train_retinanet.py --epochs 50 --batch-size 8 --lr 0.0001 --optimizer adamw
```

Hasil pelatihan akan tersimpan di folder `runs/retinanet/`:
- `best_retinanet.pth`: Bobot model dengan mAP50 tertinggi.
- `last_retinanet.pth`: Bobot checkpoint epoch terakhir.
- `training_history.csv`: Log metrik per epoch.
- `loss_curves.png`: Grafik kurva loss pelatihan & mAP.

---

## ⚖️ Cara Menjalankan Komparasi Head-to-Head

Setelah memiliki bobot `best_retinanet.pth` dan `best.pt` (YOLOv8):

```bash
cd "e:\Semester 5\Project\Egg Sorting System"

python compare_models.py \
    --yolo-weights "Egg Sorting System/models/best.pt" \
    --retina-weights "runs/retinanet/best_retinanet.pth" \
    --dataset-dir "Egg Dataset Clean" \
    --split test \
    --output-dir "runs/comparison"
```

### Output yang Dihasilkan Otomatis di `runs/comparison/`:
1. **`model_comparison_report.md`**: Laporan komparasi lengkap dengan analisis teoritis (bisa langsung disalin ke Bab 4/5 Laporan Proyek).
2. **`comparison_metrics_bar.png`**: Grafik bar perbandingan mAP50, mAP50-95, Precision, Recall, dan F1-Score.
3. **`speed_vs_accuracy.png`**: Grafik scatter plot perbandingan kecepatan (*latency*) vs akurasi (*mAP50*).
4. **`side_by_side/`**: Kumpulan citra perbandingan visual berdampingan:
   `[ Panel 1: Ground Truth ] | [ Panel 2: YOLOv8 Nano ] | [ Panel 3: RetinaNet ResNet-50 ]`

---

## 📊 Ringkasan Teoretis Komparasi untuk Laporan Semester 5

| Parameter Evaluasi | YOLOv8 Nano (`yolov8n.pt`) | RetinaNet (`ResNet-50 FPN`) |
| :--- | :---: | :---: |
| **Arsitektur Backbone** | Modified CSPDarknet53 (Ringan & Cepat) | Deep ResNet-50 + FPN (Fitur Kaya) |
| **Paradigma Deteksi** | Anchor-Free Decoupled Head | Anchor-Based Multi-scale Head |
| **Fungsi Loss** | Task-Aligned Loss (CIoU + DFL + BCE) | Focal Loss ($\alpha=0.25, \gamma=2.0$) + Smooth L1 |
| **Ukuran Bobot Model** | **~6.5 MB** | **~135 MB** (20x lipat) |
| **Total Parameter** | **~3.2 Juta Parameter** | **~34 Juta Parameter** (10x lipat) |
| **Latensi Inferensi (GPU)** | **~5 - 6 ms** per citra | **~25 - 35 ms** per citra |
| **Throughput (FPS)** | **~160 - 180 FPS** | **~30 - 40 FPS** |
| **Kelebihan Utama** | Ekstrem cepat, ringan untuk edge device (Raspberry Pi/Jetson) | Robust terhadap class imbalance ekstrem dan cacat mikro |

### Rekomendasi untuk Sistem Sortir Telur:
1. **Untuk Konveyor Real-Time (Hardware):** Gunakan **YOLOv8 Nano** sebagai model utama pada Raspberry Pi/PC konveyor karena waktu proses ~5.6 ms menjamin aktuator servo penyeleksi tidak mengalami *delay* saat telur lewat.
2. **Untuk Audit Mutu & Laboratorium:** Gunakan **RetinaNet** sebagai sistem validasi sekunder di server audit untuk memeriksa ulang telur grade tinggi atau menganalisis retakan mikro halus.

---

## 🛠️ Menggunakan RetinaNet di Aplikasi Sortir Telur

Modul adapter [`Egg Sorting System/modules/retinanet_detector.py`](file:///e:/Semester%205/Project/Egg%20Sorting%20System/Egg%20Sorting%20System/modules/retinanet_detector.py) siap diimpor ke script inspeksi Anda:

```python
from modules.retinanet_detector import RetinaNetEggDetector

# Inisialisasi detector
detector = RetinaNetEggDetector(weights_path="models/best_retinanet.pth", conf_thresh=0.40)

# Prediksi dari file gambar atau frame webcam
defect_class, confidence, detections = detector.predict("test_egg.jpg")
print(f"Hasil Deteksi: {defect_class.upper()} (Keyakinan: {confidence:.2%})")
```

