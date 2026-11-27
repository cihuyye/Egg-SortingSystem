# 🥚 Panduan Pelatihan & Komparasi Model RetinaNet vs YOLOv8 Nano
**Proyek:** Sistem Sortir Mutu Telur Otomatis Berdasarkan SNI 3926:2023  
**Modul Workshop Sistem Cerdas (Artificial Intelligence & Deep Learning)**

Repositori ini telah dilengkapi dengan *pipeline* mandiri untuk melatih, mengevaluasi, dan membandingkan model **RetinaNet (ResNet-50 FPN)** secara *head-to-head* dengan model utama **YOLOv8 Nano** pada dataset inspeksi telur independen (`Egg Dataset Clean`).

---

## 📁 Struktur Berkas Terkait RetinaNet & Komparasi

```text
Egg-SortingSystem/
├── Egg Dataset Clean/                    # Dataset bersih (717 train, 87 valid, 88 test)
│   ├── train/ (images & labels)
│   ├── valid/ (images & labels)
│   ├── test/  (images & labels)
│   └── annotations/                     # Format standar COCO JSON (hasil konversi)
│       ├── instances_train.json
│       ├── instances_valid.json
│       └── instances_test.json
├── models/
│   ├── best.pt                          # Bobot model YOLOv8 Nano utama (~6.5 MB)
│   └── best_retinanet.pth               # Bobot checkpoint RetinaNet ResNet-50 FPN (~135 MB)
├── retinanet_dataset.py                 # PyTorch Dataset loader (baca poligon YOLO & negative samples)
├── retinanet_model.py                   # Arsitektur RetinaNet ResNet-50 FPN (Torchvision Detection)
├── retinanet_metrics.py                 # Evaluator metrik mAP50, mAP50-95, Precision, Recall, F1
├── train_retinanet.py                   # Script pelatihan lokal model RetinaNet (GPU CUDA)
├── train_retinanet_cpu.py               # Script pelatihan fallback CPU (jika tidak ada GPU)
├── eval_retinanet.py                    # Script evaluasi independen pada test set
├── predict_retinanet.py                 # Script inferensi visual & deteksi cacat RetinaNet
├── compare_models.py                    # Script komparasi Head-to-Head YOLOv8 vs RetinaNet
├── yolo_to_coco.py                      # Konverter dataset YOLOv8 ke standar COCO JSON
└── README_RETINANET.md                  # Dokumentasi panduan komparasi
```

---

## 🔄 Bagaimana Dataset YOLOv8 Diadaptasi untuk RetinaNet

Dataset `Egg Dataset Clean` awalnya dibuat dalam format YOLO:
1. **Format Poligon Segmen**: Berisi baris `class_id x1 y1 x2 y2 ...` hasil ekspor LabelMe (`json_to_yolo.py`).
2. **Negative Samples**: Folder `good` berisi file `.txt` kosong berukuran 0-byte (telur utuh normal tanpa cacat).

File [`retinanet_dataset.py`](file:///d:/github_rama/Egg-SortingSystem/retinanet_dataset.py) secara otomatis:
- Menghitung bounding box pembungkus minimal `[xmin, ymin, xmax, ymax]` dari poligon:
  $$\text{xmin} = \min(x) \times W, \quad \text{xmax} = \max(x) \times W$$
  $$\text{ymin} = \min(y) \times H, \quad \text{ymax} = \max(y) \times H$$
- Memetakan ID kelas sesuai standar Torchvision Detection:
  - `0`: Background (negative sample telur normal / cangkang utuh)
  - `1`: `cracked` (cacat retak)
  - `2`: `dirt` (cacat noda kotoran)
- Menangani gambar tanpa cacat (`good`) secara aman (target berupa tensor koordinat kosong `[0, 4]`).
- Melakukan penskalaan ke $640\times 640$ piksel dan augmentasi data (*horizontal flip, brightness/contrast jitter*) secara sinkron antara citra dan bounding box (kompatibel penuh dengan Pillow 12+).

---

## 🚀 Cara Menjalankan Pelatihan RetinaNet

### Opsi 1: Latih di Komputer Lokal (Dengan GPU NVIDIA CUDA)
```bash
python retinanet/train_retinanet.py --epochs 50 --batch-size 8 --lr 0.0001 --optimizer adamw
```

### Opsi 2: Latih Menggunakan CPU (Fallback jika Tidak Ada GPU CUDA)
```bash
python retinanet/train_retinanet_cpu.py --epochs 10 --batch-size 4
```

Hasil pelatihan akan tersimpan di folder `runs/retinanet/`:
* `best_retinanet.pth`: Bobot model dengan mAP50 tertinggi.
* `last_retinanet.pth`: Bobot checkpoint epoch terakhir.
* `training_history.csv`: Log riwayat metrik per epoch.
* `loss_curves.png`: Grafik kurva loss pelatihan & mAP.

---

## ⚖️ Cara Menjalankan Komparasi Head-to-Head

Untuk mengevaluasi performa kedua model pada data uji independen (*test split*):

```bash
python compare_models.py \
    --yolo-weights "models/best.pt" \
    --retina-weights "runs/retinanet/best_retinanet.pth" \
    --dataset-dir "Egg Dataset Clean" \
    --split test \
    --output-dir "runs/comparison"
```

### Output yang Dihasilkan Otomatis di `runs/comparison/`:
1. **`model_comparison_report.md`**: Laporan komparasi kuantitatif lengkap dengan analisis teoretis (dapat langsung disalin ke bab hasil dan pembahasan laporan tugas akhir).
2. **`comparison_metrics_bar.png`**: Grafik bar perbandingan mAP50, mAP50-95, Precision, Recall, dan F1-Score.
3. **`speed_vs_accuracy.png`**: Grafik scatter plot perbandingan kecepatan (*latency*) vs akurasi (*mAP50*).
4. **`side_by_side/`**: Kumpulan citra visualisasi perbandingan berdampingan:
   `[ Panel 1: Ground Truth ] | [ Panel 2: YOLOv8 Nano ] | [ Panel 3: RetinaNet ResNet-50 ]`

---

## 📊 Ringkasan Teoretis Komparasi untuk Laporan Sistem Cerdas

| Parameter Evaluasi | YOLOv8 Nano (`best.pt`) | RetinaNet (`best_retinanet.pth`) |
| :--- | :---: | :---: |
| **Arsitektur Backbone** | Modified CSPDarknet53 (Ringan & Cepat) | Deep ResNet-50 + FPN (Fitur Kaya) |
| **Paradigma Deteksi** | Anchor-Free Decoupled Head | Anchor-Based Multi-scale Head |
| **Fungsi Loss** | Task-Aligned Loss (CIoU + DFL + BCE) | Focal Loss ($\alpha=0.25, \gamma=2.0$) + Smooth L1 |
| **Ukuran Bobot Model** | **~6.5 MB** | **~135 MB** (~20x lebih besar) |
| **Total Parameter** | **~3.2 Juta Parameter** | **~34 Juta Parameter** (~10x lebih banyak) |
| **Latensi Inferensi (GPU)** | **~5 - 6 ms** per citra | **~25 - 35 ms** per citra |
| **Throughput (FPS)** | **~160 - 180 FPS** | **~30 - 40 FPS** |
| **Karakteristik Utama** | Inferensi super cepat, sangat efisien untuk edge device konveyor | Robust terhadap ketidakseimbangan kelas ekstrem dan cacat mikro |

---

## 🎯 Rekomendasi Pemilihan Model untuk Sistem Industri

1. **Untuk Konveyor Real-Time (Hardware):** Gunakan **YOLOv8 Nano** sebagai model utama pada konveyor karena waktu proses $\sim 5\text{ ms}$ menjamin aktuator servo/pneumatik penyeleksi tidak mengalami keterlambatan (*zero-delay actuation*) saat telur melintas.
2. **Untuk Audit Mutu Laboratorium:** Gunakan **RetinaNet** sebagai sistem verifikasi kedua di server audit untuk menginspeksi ulang sampel telur bernilai tinggi atau menganalisis retakan mikro rambut yang sangat halus.
