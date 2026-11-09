# Egg Dataset Clean (Bebas Data Leakage)

Dataset ini dibuat melalui proses **Tahap 1 Audit Dataset**:
1. Menghapus seluruh duplikasi augmentasi Roboflow (.rf.) yang sebelumnya bocor antar split train/valid/test.
2. Setiap butir telur fisik unik hanya dialokasikan ke **SATU** folder (Strict Patient/Instance Level Split).
3. Menggunakan Stratified Sampling (70% Train, 15% Valid, 15% Test) dengan seed deterministik (seed=42).

## Distribusi Citra Bersih

| Kelas | Train | Valid | Test | Total Telur Fisik | Persentase |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **cracked** | 122 | 26 | 26 | **174** | 29.8% |
| **dirt** | 225 | 48 | 48 | **321** | 55.0% |
| **good** | 62 | 13 | 14 | **89** | 15.2% |
| **Total** | **409** | **87** | **88** | **584** | 100.0% |

## Verifikasi Kebocoran Data (Data Leakage Check)
- Train vs Valid overlap: 0 citra (0.0%)
- Train vs Test overlap: 0 citra (0.0%)
- Valid vs Test overlap: 0 citra (0.0%)

## Tahap 2: Penyeimbangan Kelas & Augmentasi Warna Cangkang
Untuk menghilangkan bias warna cangkang (*confounder*) dan menyeimbangkan data latih:
- **`good` (Train):** Diperluas dengan 3 variasi warna alami (Light Brown, Golden Tan, Cream). Total: 248 citra.
- **`cracked` (Train):** Diperluas dengan variasi Horizontal Flip. Total: 244 citra.
- **`dirt` (Train):** 225 citra asli dipertahankan tanpa perubahan.
- **`valid` & `test`:** Tetap 100% citra asli independen (murni tanpa augmentasi sintetis).

Rasio Kelas pada Train Set sekarang **seimbang ~1:1:1** (~33% per kelas).
