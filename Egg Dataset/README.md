# Dataset Telur — Sudah Dipreprocessing

Ini folder dataset foto telur yang sudah beres dipreprocessing dan langsung bisa dipakai buat training YOLOv8. Semua gambar sudah seragam ukurannya, background-nya sudah hitam, dan label bounding box-nya sudah lengkap.

## Kelas Label

Pakai format label standar YOLO (satu file .txt per gambar):
- `0` = `cracked` — cangkang retak, termasuk retakan rambut yang tipis
- `1` = `dirt` — cangkang kotor, ada noda kotoran ayam atau bercak
- `2` = `good` — cangkang bersih, mulus, ga ada cacat

Tiap bounding box dibuat melingkupi satu butir telur utuh.

## Apa Aja yang Sudah Dilakukan Saat Preprocessing

1. **Resize ke 640x640 pakai Lanczos** — Kenapa Lanczos? Karena metode ini bisa jaga ketajaman garis retakan halus di cangkang. Kalau pakai resize biasa (bilinear), garis retak rambut yang cuma 1-2 piksel lebarnya bisa ikut kabur dan ga kedeteksi sama model.
2. **Ganti Background Jadi Hitam** — Latar belakang foto aslinya macam-macam (ada meja, kain, wadah telur). Semuanya diganti hitam pekat biar mirip kondisi di dalam kotak pemindai (dark box) di atas konveyor nanti.
3. **Anti-Aliasing di Pinggiran Telur** — Abis background dipotong, pinggiran telur suka jadi bergerigi. Dikasih supersampling 2x dan Gaussian feathering biar halus lagi.
4. **Cek Ulang Pasangan File** — Semua file gambar (.jpg) dipastikan punya file label (.txt) yang cocok. Ga ada yang nyasar atau kehilangan pasangan.

## Pembagian Data

Pembagiannya pakai stratified split supaya proporsi tiap kelas tetap seimbang di setiap bagian:

| Folder | Cracked | Dirt | Good | Total | Keterangan |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **train** | 249 | 225 | 88 | 562 | Buat pelatihan model (~70%) |
| **valid** | 71 | 64 | 25 | 160 | Buat validasi waktu training (~20%) |
| **test** | 36 | 32 | 12 | 80 | Buat tes performa akhir (~10%) |
| **Total** | **356** | **321** | **125** | **802** | Semua punya label lengkap |

## Struktur Folder

```text
Egg Dataset/
├── data.yaml              # Setting path dan nama kelas buat YOLOv8
├── README.md
├── train/
│   ├── images/ (cracked, dirt, good)
│   └── labels/ (cracked, dirt, good)
├── valid/
│   ├── images/ (cracked, dirt, good)
│   └── labels/ (cracked, dirt, good)
└── test/
    ├── images/ (cracked, dirt, good)
    └── labels/ (cracked, dirt, good)
```

## Isi data.yaml

```yaml
train: train/images
val: valid/images
test: test/images

nc: 3
names: ['cracked', 'dirt', 'good']
```
