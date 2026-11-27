# 🚀 YOLOv26 Nano Instance Segmentation (`yolo26n-seg`)

Direktori ini berisi seluruh artefak pelatihan, notebook Google Colab, matriks evaluasi, dan bobot checkpoint untuk model generasi terbaru **YOLOv26-seg** (*NMS-Free End-to-End Object Detection & Instance Segmentation*) untuk sistem inspeksi mutu telur **SNI 3926:2023**.

---

## 📁 Struktur Direktori `yolov26/`

```text
yolov26/
├── Train_YOLOv26_Segmentation_Colab.ipynb  # Notebook Colab pelatihan GPU T4
├── train_yolo26.py                        # Script pelatihan lokal (CUDA/CPU)
├── predict_yolo26.py                      # Script inferensi & visualisasi mask
├── weights/
│   └── best.pt                           # Bobot model terbaik YOLOv26-seg
├── matriks/                              # Seluruh kurva evaluasi (Loss, PR, F1, dll.)
├── __init__.py
└── README.md
```

---

## ⚡ Keunggulan Arsitektur YOLOv26 pada Kasus Telur

1. **NMS-Free (Eliminasi Non-Maximum Suppression)**:
   - Output prediksi keluar langsung dari neural network secara *end-to-end* tanpa penyaringan kotak berulang di CPU.
   - Menghasilkan latensi inferensi yang sangat stabil dan konstan pada konveyor bergerak.
2. **STAL (Small-Target-Aware Assignment)**:
   - Mengoptimalkan alokasi gradien untuk mendeteksi retakan mikro rambut cangkang (*hairline cracks*) dan bintik noda mikro.
3. **MuSGD Optimizer & ProgLoss**:
   - Mempercepat konvergensi pelatihan dan meminimalkan resiko *gradient explosion* pada arsitektur Nano.

---

## 🛠️ Cara Melatih di Google Colab

1. Buka [Google Colab](https://colab.research.google.com).
2. Unggah notebook `Train_YOLOv26_Segmentation_Colab.ipynb`.
3. Atur Runtime ke **T4 GPU** (*Runtime* -> *Change runtime type* -> **T4 GPU**).
4. Unggah `Egg_Dataset_Clean.zip` ke panel Files.
5. Jalankan Cell 1 hingga Cell 7. File `best.pt` dan arsip grafik akan terunduh otomatis.
