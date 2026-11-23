"""
================================================================================
SKRIP DEMONSTRASI & PEMBUKTIAN EKSTRAKSI FITUR (PENGOLAHAN CITRA & VISION)
Workshop Pengolahan Citra dan Vision - Sistem Sortir Telur SNI 3926:2023
================================================================================
Skrip ini membuktikan seluruh tahapan pipeline Computer Vision dari:
1. Citra Input (Akuisisi Kamera)
2. Preprocessing (Grayscale, Blurring, HSV Analysis)
3. Segmentasi Objek (Otsu Thresholding & Operasi Morfologi)
4. Ekstraksi Fitur Geometri, Morfologi, Warna, & Estimasi Fisika
5. Output Vektor Fitur Numerik & Visualisasi Hasil Overlay
================================================================================
"""

import os
import sys
import cv2
import numpy as np

# Tambahkan direktori root ke sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from modules.shape_size_detector import measure_egg_shape_and_size


def run_feature_extraction_demo(image_path: str, output_image_path: str = os.path.join("hasil_pembuktian", "bukti_ekstraksi_fitur.png")):
    print("=" * 80)
    print("  PEMBUKTIAN EKSTRAKSI FITUR PENGOLAHAN CITRA & COMPUTER VISION")
    print("  Standar Acuan: SNI 3926:2023 (Telur Ayam Konsumsi)")
    print("=" * 80)

    if not os.path.exists(image_path):
        print(f"❌ File gambar tidak ditemukan: {image_path}")
        return

    # 1. BACA CITRA ASLI (Input Akuisisi)
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"❌ File gambar tidak dapat dibaca: {image_path}")
        return

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h_orig, w_orig = img_bgr.shape[:2]
    print(f"\n[1] AKUISISI CITRA INPUT:")
    print(f"    - Dimensi Citra    : {w_orig} x {h_orig} piksel (3 Channel BGR)")

    # 2. PREPROCESSING: Normalisasi, Grayscale, Gaussian Blur
    standard_dim = 640.0
    scale = standard_dim / max(h_orig, w_orig)
    new_w, new_h = int(w_orig * scale), int(h_orig * scale)
    img_scaled = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(img_scaled, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    print(f"\n[2] PREPROCESSING & FILTERING:")
    print(f"    - Skala Normalisasi : {scale:.4f} (Canvas 640px)")
    print(f"    - Color Conversion  : BGR -> Grayscale")
    print(f"    - Spatial Filtering : Gaussian Blur 7x7 (Reduksi Noise)")

    # 3. SEGMENTASI CITRA: Otsu Thresholding + Operasi Morfologi
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    corners = [float(gray[0, 0]), float(gray[0, -1]), float(gray[-1, 0]), float(gray[-1, -1])]
    if np.mean(corners) > 100:
        thresh = cv2.bitwise_not(thresh)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    print(f"\n[3] SEGMENTASI & MORFOLOGI:")
    print(f"    - Binarization      : Adaptive Otsu Thresholding")
    print(f"    - Operasi Morfologi : Closing (Kernel Elips 5x5) untuk mengisi lubang")

    # 4. EKSTRAKSI KONTUR & CONVEX HULL
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("❌ Tidak ada kontur yang terdeteksi.")
        return

    best_cnt = max(contours, key=cv2.contourArea)
    area_px = cv2.contourArea(best_cnt)
    perimeter_px = cv2.arcLength(best_cnt, True)
    hull = cv2.convexHull(best_cnt)
    hull_area = cv2.contourArea(hull)
    solidity = area_px / max(1.0, hull_area)

    # 5. FIT ELLIPSE & PENGUKURAN DIMENSI
    ellipse = cv2.fitEllipse(best_cnt)
    (cx_s, cy_s), (ax1_s, ax2_s), angle = ellipse
    minor_px = min(ax1_s, ax2_s)
    major_px = max(ax1_s, ax2_s)

    base_px_to_mm = 0.135
    length_mm = major_px * base_px_to_mm
    width_mm = minor_px * base_px_to_mm
    esi = (width_mm / length_mm) * 100.0 if length_mm > 0 else 0

    # Estimasi volume elipsoid & bobot
    vol_cm3 = (np.pi / 6.0) * (length_mm / 10.0) * ((width_mm / 10.0) ** 2)
    weight_gram = vol_cm3 * 1.035

    # 6. EKSTRAKSI FITUR WARNA HSV & TEKSTUR
    mask_cnt = np.zeros_like(gray)
    cv2.drawContours(mask_cnt, [best_cnt], -1, 255, -1)
    hsv_scaled = cv2.cvtColor(img_scaled, cv2.COLOR_BGR2HSV)
    mean_h, mean_s, mean_v = cv2.mean(hsv_scaled, mask=mask_cnt)[:3]
    _, std_bgr = cv2.meanStdDev(img_scaled, mask=mask_cnt)
    texture_uniformity = np.mean(std_bgr)

    # 7. TABEL VEKTOR FITUR (NUMERICAL PROOF)
    print("\n" + "=" * 80)
    print("  VEKTOR FITUR YANG DIEKSTRAKSI (BUKTI NUMERIK)")
    print("=" * 80)
    print(f"{'No':<4} | {'Nama Fitur':<28} | {'Nilai Terhitung':<18} | {'Metode / Rumus':<24}")
    print("-" * 80)
    print(f"{'1':<4} | {'Luas Kontur (Area)':<28} | {area_px:<18.1f} | {'cv2.contourArea (px)'}")
    print(f"{'2':<4} | {'Keliling (Perimeter)':<28} | {perimeter_px:<18.1f} | {'cv2.arcLength (px)'}")
    print(f"{'3':<4} | {'Convex Hull Area':<28} | {hull_area:<18.1f} | {'cv2.convexHull (px)'}")
    print(f"{'4':<4} | {'Solidity (Kecembungan)':<28} | {solidity:<18.4f} | {'Area / Hull Area'}")
    print(f"{'5':<4} | {'Major Axis (Panjang)':<28} | {length_mm:<18.2f} mm | {'FitEllipse * px_to_mm'}")
    print(f"{'6':<4} | {'Minor Axis (Lebar)':<28} | {width_mm:<18.2f} mm | {'FitEllipse * px_to_mm'}")
    print(f"{'7':<4} | {'Egg Shape Index (ESI)':<28} | {esi:<18.2f} %  | {'(W / L) * 100 (SNI)'}")
    print(f"{'8':<4} | {'Estimasi Volume':<28} | {vol_cm3:<18.2f} cm³ | {'(pi/6) * L * W²'}")
    print(f"{'9':<4} | {'Estimasi Bobot':<28} | {weight_gram:<18.2f} gram| {'Volume * 1.035 g/cm³'}")
    print(f"{'10':<4} | {'Warna: Rata-rata Hue':<28} | {mean_h:<18.1f} | {'cv2.cvtColor HSV (H)'}")
    print(f"{'11':<4} | {'Warna: Rata-rata Saturation':<28} | {mean_s:<18.1f} | {'cv2.cvtColor HSV (S)'}")
    print(f"{'12':<4} | {'Warna: Rata-rata Value':<28} | {mean_v:<18.1f} | {'cv2.cvtColor HSV (V)'}")
    print(f"{'13':<4} | {'Standar Deviasi Tekstur':<28} | {texture_uniformity:<18.2f} | {'cv2.meanStdDev (RGB)'}")
    print("=" * 80)

    # Status SNI
    shape_status = "Normal (72-76%)" if (72.0 <= esi <= 76.0) else ("Lonjong (<72%)" if esi < 72 else "Bulat (>76%)")
    weight_cat = "Kecil (<50g)" if weight_gram < 50 else ("Sedang (50-60g)" if weight_gram <= 60 else "Besar (>60g)")
    print(f"\n[HASIL KELAS MORFOLOGI SNI 3926:2023]:")
    print(f"  • Kategori Bentuk (ESI) : {shape_status}")
    print(f"  • Kategori Ukuran Bobot : {weight_cat}")

    # 8. GENERATE PANEL VISUALISASI 6 TAHAP
    # Panel 1: Citra Asli
    p1 = img_scaled.copy()
    cv2.putText(p1, "1. Citra Asli", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Panel 2: Grayscale
    p2 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.putText(p2, "2. Grayscale & Blur", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Panel 3: Biner & Morfologi
    p3 = cv2.cvtColor(morph, cv2.COLOR_GRAY2BGR)
    cv2.putText(p3, "3. Biner Otsu + Morfologi", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Panel 4: Kontur & Convex Hull
    p4 = img_scaled.copy()
    cv2.drawContours(p4, [best_cnt], -1, (0, 255, 0), 2)
    cv2.drawContours(p4, [hull], -1, (0, 0, 255), 2)
    cv2.putText(p4, f"4. Kontur (Hijau) & Hull (Merah)", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(p4, f"Solidity: {solidity:.3f}", (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Panel 5: Fit Ellipse & Sumbu Geometri
    p5 = img_scaled.copy()
    cv2.ellipse(p5, ellipse, (255, 0, 0), 2)
    # Gambar sumbu mayor & minor
    rad = np.deg2rad(angle)
    p_center = (int(cx_s), int(cy_s))
    dx_maj = int((major_px / 2.0) * np.sin(rad))
    dy_maj = int(-(major_px / 2.0) * np.cos(rad))
    dx_min = int((minor_px / 2.0) * np.cos(rad))
    dy_min = int((minor_px / 2.0) * np.sin(rad))
    cv2.line(p5, (p_center[0] - dx_maj, p_center[1] - dy_maj), (p_center[0] + dx_maj, p_center[1] + dy_maj), (0, 0, 255), 2)
    cv2.line(p5, (p_center[0] - dx_min, p_center[1] - dy_min), (p_center[0] + dx_min, p_center[1] + dy_min), (255, 255, 0), 2)
    cv2.putText(p5, f"5. Fit Ellipse & Axes", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    cv2.putText(p5, f"L:{length_mm:.1f}mm W:{width_mm:.1f}mm", (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # Panel 6: Hasil Akhir Ekstraksi Fitur Overlay
    p6 = img_scaled.copy()
    cv2.ellipse(p6, ellipse, (0, 255, 0), 2)
    box_w, box_h = 320, 160
    overlay = p6.copy()
    cv2.rectangle(overlay, (10, 10), (10 + box_w, 10 + box_h), (0, 0, 0), -1)
    p6 = cv2.addWeighted(overlay, 0.65, p6, 0.35, 0)
    cv2.putText(p6, "6. HASIL EKSTRAKSI FITUR", (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    cv2.putText(p6, f"Panjang (L) : {length_mm:.2f} mm", (20, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(p6, f"Lebar (W)   : {width_mm:.2f} mm", (20, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(p6, f"Shape Index : {esi:.2f}% ({shape_status})", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(p6, f"Est. Bobot  : {weight_gram:.2f}g ({weight_cat})", (20, 122), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.putText(p6, f"Solidity    : {solidity:.3f} | Area: {int(area_px)}px", (20, 144), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    # Gabungkan 6 panel menjadi 2 baris x 3 kolom
    row1 = np.hstack([p1, p2, p3])
    row2 = np.hstack([p4, p5, p6])
    composite = np.vstack([row1, row2])

    output_full_path = os.path.join(BASE_DIR, output_image_path)
    cv2.imwrite(output_full_path, composite)
    print(f"\n✅ Gambar bukti 6 tahap berhasil disimpan ke:")
    print(f"   -> {output_full_path}")
    print("=" * 80)


if __name__ == "__main__":
    # Cari sampel citra yang ada di dataset
    sample_img = os.path.join(BASE_DIR, "Egg Dataset Clean", "test", "images", "cracked", "105_png_flip.jpg")
    if not os.path.exists(sample_img):
        # Fallback ke raw jika ada
        raw_dir = os.path.join(BASE_DIR, "Egg Dataset (Raw)")
        if os.path.exists(raw_dir):
            for r, d, f in os.walk(raw_dir):
                for file in f:
                    if file.lower().endswith(('.jpg', '.png')):
                        sample_img = os.path.join(r, file)
                        break
                if os.path.exists(sample_img):
                    break

    run_feature_extraction_demo(sample_img, os.path.join("hasil_pembuktian", "bukti_ekstraksi_fitur.png"))

