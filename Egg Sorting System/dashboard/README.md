# Egg Dashboard (Inspeksi Cangkang 360° & Grading SNI 3926:2023)

Web dashboard visual untuk demonstrasi sistem grading telur berbasis **YOLOv8** (deteksi cacat cangkang multi-sudut) dan **OpenCV** (geometri, bentuk, & estimasi ukuran) dengan backend **FastAPI**.

---

### Fitur Utama:
1. **Inspeksi Cangkang 360° (Multi-Angle View)**:
   - Pengguna dapat mengunggah **1 atau beberapa foto** (misal: Sisi Depan, Sisi Belakang, Sisi Samping) untuk **1 butir telur yang sama**.
   - Sistem YOLOv8 mengevaluasi setiap sudut dan menerapkan aturan keselamatan mutu:
     - Jika ada **satu saja sudut yang retak (*cracked*)**, telur langsung divonis **Mutu III (Reject)**.
     - Jika ada noda kotoran cangkang (*dirt*), telur masuk **Mutu II (Grade B)**.
     - Hanya jika **seluruh sudut cangkang 360° bersih dan utuh**, telur dapat meraih **Mutu I (Grade A)**.
2. **OpenCV Geometri & Ukuran Fisik**:
   - Menghitung dimensi panjang & lebar (mm).
   - Menghitung rasio *Egg Shape Index* (ESI %) sesuai standar normal SNI (72% - 76%).
   - Estimasi bobot (gram) dan kategori ukuran SNI (Ekstra Besar, Besar, Sedang, Kecil).
3. **Tanpa Modul Candling**:
   - Modul candling dinonaktifkan sementara dari dashboard agar fokus pada data citra cangkang dan bentuk telur yang nyata tersedia.
4. **Tema Visual Cerah (*Bright & Cheerful*)**:
   - Dirancang rapi, modern, dan profesional untuk demo presentasi.

---

### Cara Menjalankan Lewat Terminal:

1. Buka terminal di folder proyek `D:\Egg Sorting System`.
2. Jalankan server FastAPI:
   ```bash
   python app_api.py
   ```
3. Tekan **`Ctrl` + Klik link `http://127.0.0.1:8000`** yang muncul di terminal.
4. Dashboard akan langsung terbuka di browser dan terhubung ke API.
5. Unggah 1 atau beberapa foto telur dari `D:\Egg Dataset\Egg Dataset\test\images\` lalu klik **"⚡ Analisis 360° & Tentukan Grade"**.
