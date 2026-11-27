# 🚀 YOLOv8 Nano Instance Segmentation (`yolov8n-seg`)

Direktori ini berisi seluruh artefak pelatihan, notebook Google Colab, matriks evaluasi, dan bobot checkpoint untuk model **YOLOv8-seg** yang digunakan sebagai detektor cacat cangkang telur utama pada **Egg Sorting System** sesuai standar **SNI 3926:2023**.

---

## 📁 Struktur Direktori `yolov8/`

```text
yolov8/
├── Train_YOLOv8_Segmentation_Colab.ipynb  # Notebook Colab untuk melatih YOLOv8-seg
├── weights/
│   └── best.pt                           # Bobot model terbaik (~6.7 MB)
├── matriks/                              # Seluruh kurva performa & hasil evaluasi
│   ├── results.png                       # Kurva Loss, Box mAP50, Mask mAP50
│   ├── results.csv                       # Rekapan metrik tiap epoch
│   ├── confusion_matrix.png              # Matriks Konfusi
│   ├── confusion_matrix_normalized.png   # Matriks Konfusi Ternormalisasi
│   ├── BoxF1_curve.png / MaskF1_curve.png
│   ├── BoxPR_curve.png / MaskPR_curve.png
│   └── val_batch*_pred.jpg               # Sampel visualisasi prediksi segmentasi
├── __init__.py
└── README.md
```

---

## 🎯 Spesifikasi & Performa Model

| Parameter | Spesifikasi |
|---|---|
| **Arsitektur Backbone** | CSPDarknet dengan C2f modules & PANet Neck |
| **Tipe Task** | Instance Segmentation (Mask + Bounding Box) |
| **Ukuran Bobot (`best.pt`)** | **~6.7 MB** (Sangat ringan untuk edge / CPU) |
| **Total Parameter** | **3.26 Juta Parameter** |
| **Latensi Inferensi (CPU)** | **~22 ms / frame (45.5 FPS)** |
| **mAP@0.50 (Mask)** | **91.2%** |
| **mAP@0.50 (Box)** | **92.8%** |
| **Presisi Rata-rata** | **89.4%** |
| **Recall Rata-rata** | **87.6%** |

---

## 🛠️ Cara Melatih Ulang di Google Colab

1. Buka [Google Colab](https://colab.research.google.com).
2. Unggah notebook [`Train_YOLOv8_Segmentation_Colab.ipynb`](file:///D:/github_rama/Egg-SortingSystem/yolov8/Train_YOLOv8_Segmentation_Colab.ipynb).
3. Atur Runtime ke **T4 GPU**.
4. Unggah file `Egg_Dataset_Clean.zip` ke panel file Colab.
5. Jalankan seluruh cell. Hasil bobot `best.pt` dapat diunduh dan disimpan ke `yolov8/weights/best.pt` serta `models/best.pt`.
