from typing import Any, Dict, Optional, Tuple, Union
import cv2
import numpy as np

def validate_egg_candidate(img_rgb: np.ndarray, cnt: np.ndarray, total_area: int) -> Tuple[bool, Optional[Tuple], float]:
    """
    Validasi ilmiah tingkat tinggi morfologi cangkang telur ayam konsumsi:
    1. Filter Tepi Frame: Menolak kontur yang terpotong di tepi layar kamera (misal tangan/meja/benda asing).
    2. Proporsi Luas Kontur: 2.0% s/d 85% luas frame.
    3. Rasio Aspek Elips: 0.55 - 0.90 (menolak lingkaran sempurna seperti bola/tutup botol dan objek pipih).
    4. Solidity (Kecembungan cangkang): >= 0.85 (bebas lekukan tajam, siku, atau sudut lancip).
    5. Validasi Spektrum Warna HSV: Mendeteksi rentang kromatis cangkang cokelat ras atau putih gading.
    6. Keseragaman Tekstur: Menolak objek berpola ramai, teks, atau gambar bertekstur bising.
    """
    h_orig, w_orig = img_rgb.shape[:2]
    area = cv2.contourArea(cnt)
    if area < (0.018 * total_area) or area > (0.85 * total_area) or len(cnt) < 5:
        return False, None, 0.0

    # 1. FILTER TEPI LAYAR: Tolak kontur yang menyentuh pinggir layar kamera (misal tangan/jari/objek terpotong)
    x_min, y_min = np.min(cnt[:, 0, 0]), np.min(cnt[:, 0, 1])
    x_max, y_max = np.max(cnt[:, 0, 0]), np.max(cnt[:, 0, 1])
    if x_min <= 2 or y_min <= 2 or x_max >= w_orig - 3 or y_max >= h_orig - 3:
        return False, None, 0.0

    try:
        ellipse = cv2.fitEllipse(cnt)
        (cx, cy), (ax1, ax2), angle = ellipse
        minor = min(ax1, ax2)
        major = max(ax1, ax2)

        if minor <= 0 or major <= 0:
            return False, None, 0.0

        # 2. RASIO ASPEK: Telur tidak pernah bulat sempurna (> 0.90) dan tidak pipih (< 0.55)
        aspect_ratio = minor / major
        if not (0.55 <= aspect_ratio <= 0.90):
            return False, None, 0.0

        # 3. SOLIDITY: Perbandingan luas kontur terhadap convex hull (kecembungan)
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        if hull_area <= 0:
            return False, None, 0.0
        solidity = area / hull_area
        if solidity < 0.84:
            return False, None, 0.0

        # 4. VALIDASI WARNA KROMATIK HSV & RGB CANGKANG TELUR
        mask_single = np.zeros((h_orig, w_orig), dtype=np.uint8)
        cv2.drawContours(mask_single, [cnt], -1, 255, -1)

        hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        h_val, s_val, v_val = cv2.mean(hsv, mask=mask_single)[:3]

        # Cangkang cokelat ras: Hue 3-38 (kuning-oranye-cokelat), Saturasi 16-205, Kecerahan >= 40
        # Cangkang putih: Saturasi rendah <= 45, Kecerahan tinggi >= 95
        is_brown_egg = (3 <= h_val <= 38) and (16 <= s_val <= 205) and (v_val >= 40)
        is_white_egg = (s_val <= 45) and (v_val >= 95)

        if not (is_brown_egg or is_white_egg):
            return False, None, 0.0

        # 5. KESERAGAMAN TEKSTUR: Menolak objek bercorak ramai / bergambar / wajah / teks
        _, std_rgb = cv2.meanStdDev(img_rgb, mask=mask_single)
        if np.mean(std_rgb) > 58.0:
            return False, None, 0.0

        score = area * solidity
        return True, ellipse, score

    except Exception:
        return False, None, 0.0


def measure_egg_shape_and_size(image_np: np.ndarray, base_px_to_mm: float = 0.135) -> Optional[Dict[str, Any]]:
    """
    Mengukur dimensi panjang, lebar, dan Egg Shape Index (ESI/SI) telur
    berdasarkan standar SNI 3926:2023 (Tabel 1 & Pasal 4.2).
    Dilengkapi filter anti-objek-palsu (validasi morfologi cangkang & batas fisik wajar).
    """
    h_orig, w_orig = image_np.shape[:2]
    if h_orig < 60 or w_orig < 60:
        return None

    # 1. Normalisasi resolusi citra ke ukuran standar konveyor (640px)
    standard_dim = 640.0
    scale_factor = standard_dim / max(h_orig, w_orig)
    new_w = max(10, int(w_orig * scale_factor))
    new_h = max(10, int(h_orig * scale_factor))
    img_scaled = cv2.resize(image_np, (new_w, new_h), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(img_scaled, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # Adaptive Otsu Thresholding
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    corners = [float(gray[0, 0]), float(gray[0, -1]), float(gray[-1, 0]), float(gray[-1, -1])]
    is_bright_bg = np.mean(corners) > 100
    if is_bright_bg:
        thresh = cv2.bitwise_not(thresh)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Cari kontur telur yang valid secara ilmiah
    valid_candidates = []
    scaled_total_area = new_h * new_w
    for cnt in contours:
        ok, ellipse, score = validate_egg_candidate(img_scaled, cnt, scaled_total_area)
        if ok and ellipse is not None:
            valid_candidates.append((cnt, ellipse, score))

    if not valid_candidates:
        return None

    best_cnt, best_ellipse_scaled, _ = max(valid_candidates, key=lambda c: c[2])
    (cx_s, cy_s), (ax1_s, ax2_s), angle = best_ellipse_scaled

    minor_axis = min(ax1_s, ax2_s)  # Lebar dalam skala 640px
    major_axis = max(ax1_s, ax2_s)  # Panjang dalam skala 640px

    length_mm = major_axis * base_px_to_mm
    width_mm = minor_axis * base_px_to_mm

    # Batas fisik telur ayam konsumsi (SNI 3926:2023):
    # Panjang wajar 38 - 75 mm, Lebar wajar 28 - 55 mm
    if not (36.0 <= length_mm <= 78.0) or not (26.0 <= width_mm <= 58.0):
        return None

    # Rumus SNI 3926:2023 Lampiran A.1.1.3.1: SI = (W / L) * 100
    si = (width_mm / length_mm) * 100.0 if length_mm > 0 else 0
    if not (58.0 <= si <= 94.0):
        return None

    # Standar SNI 3926:2023 Tabel 1: Normal = SI 72 sampai 76.
    if 72.0 <= si <= 76.0:
        shape_status = "Normal (72-76%)"
    elif 76.0 < si <= 80.0:
        shape_status = "Agak Bulat (76-80%)"
    elif si > 80.0:
        shape_status = "Bulat (> 80%)"
    else:
        shape_status = "Lonjong (< 72%)"

    # Estimasi volume elipsoid: V = (PI / 6) * L * W^2 (cm^3)
    # Massa jenis rata-rata telur segar ~ 1.035 g/cm^3
    l_cm = length_mm / 10.0
    w_cm = width_mm / 10.0
    vol_cm3 = (np.pi / 6.0) * l_cm * (w_cm ** 2)
    weight_gram = vol_cm3 * 1.035

    # Batas bobot wajar telur ayam konsumsi (SNI 3926:2023): 28 - 85 gram
    if not (26.0 <= weight_gram <= 88.0):
        return None

    # Klasifikasi Bobot SNI 3926:2023 Pasal 4.2
    if weight_gram < 50.0:
        size_category = "Kecil (< 50g)"
    elif 50.0 <= weight_gram <= 60.0:
        size_category = "Sedang (50g - 60g)"
    elif 60.0 < weight_gram <= 65.0:
        size_category = "Besar (60g - 65g)"
    else:
        size_category = "Ekstra Besar (> 65g)"

    # Petakan kembali elips ke koordinat citra asli
    orig_ellipse = (
        (float(cx_s / scale_factor), float(cy_s / scale_factor)),
        (float(ax1_s / scale_factor), float(ax2_s / scale_factor)),
        float(angle)
    )

    return {
        "is_valid_egg": True,
        "length_mm": round(length_mm, 2),
        "width_mm": round(width_mm, 2),
        "esi_percentage": round(si, 2),
        "shape_status": shape_status,
        "estimated_weight_gram": round(weight_gram, 2),
        "size_category": size_category,
        "standard_reference": "SNI 3926:2023 Tabel 1 & Pasal 4.2",
        "ellipse_orig": orig_ellipse
    }


def get_clean_egg_canvas(img_rgb: np.ndarray, ellipse_orig: Optional[Tuple] = None, target_size: int = 640):
    """
    Menyiapkan citra untuk YOLOv8 dengan latar belakang hitam pekat (domain konveyor):
    - Jika latar belakang terang/ubin lantai/meja kayu, masking telur secara presisi
      menggunakan elips cangkang dan bersihkan semua nat keramik/bayangan/pola lantai ke hitam [0,0,0].
    - Resize dan tempatkan di tengah canvas 640x640 tanpa distorsi rasio aspek.
    - Mengembalikan canvas RGB 640x640, skala, x_off, y_off.
    """
    h_orig, w_orig = img_rgb.shape[:2]
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    corners = [float(gray[0, 0]), float(gray[0, -1]), float(gray[-1, 0]), float(gray[-1, -1])]
    is_bright_bg = np.mean(corners) > 85

    # Jika latar belakang terang atau lantai keramik dan elips tersedia: mask latar luar
    if is_bright_bg and ellipse_orig is not None:
        egg_mask = np.zeros((h_orig, w_orig), dtype=np.uint8)
        cv2.ellipse(egg_mask, ellipse_orig, 255, -1)
        egg_mask = cv2.GaussianBlur(egg_mask, (7, 7), 0)
        mask_3ch = np.repeat(egg_mask[:, :, np.newaxis] / 255.0, 3, axis=2)
        clean_img = (img_rgb * mask_3ch).astype(np.uint8)
    else:
        clean_img = img_rgb

    scale = 480.0 / max(h_orig, w_orig)
    nh, nw = int(h_orig * scale), int(w_orig * scale)
    resized = cv2.resize(clean_img, (nw, nh), interpolation=cv2.INTER_LANCZOS4)

    canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    y_off = (target_size - nh) // 2
    x_off = (target_size - nw) // 2
    canvas[y_off:y_off + nh, x_off:x_off + nw] = resized

    return canvas, scale, x_off, y_off


def preprocess_egg_image(
    img_rgb: np.ndarray,
    target_canvas: int = 640,
    target_egg_height: float = 416.0,
    return_transform: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, Dict]]:
    """
    Preprocessing citra telur agar sesuai dengan domain data latih YOLOv8:
    - Segmentasi telur via HSV saturation + Otsu threshold + hole-fill
    - fitEllipse untuk mendapatkan orientasi dan ukuran telur
    - Affine transform: skala telur ke target_egg_height px, center di tengah canvas
    - Anti-aliased feathering mask (supersampled) + background normalisasi hitam
    - Mengembalikan canvas RGB (target_canvas x target_canvas)
    - Jika return_transform=True, mengembalikan juga dict transform_info
    """
    h_orig, w_orig = img_rgb.shape[:2]

    # 1. Segmentasi telur: HSV saturation channel + Otsu
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1]

    # Otsu threshold pada channel saturasi
    _, sat_mask = cv2.threshold(sat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Cek apakah latar belakang terang (telur lebih saturated dari background)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    corners = [float(gray[0, 0]), float(gray[0, -1]), float(gray[-1, 0]), float(gray[-1, -1])]
    is_bright_bg = np.mean(corners) > 100

    # Fallback ke grayscale Otsu jika saturasi gagal
    _, gray_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if is_bright_bg:
        gray_mask = cv2.bitwise_not(gray_mask)

    # Gabungkan kedua mask
    combined_mask = cv2.bitwise_or(sat_mask, gray_mask)

    # Hole-fill: morphological close + open
    kernel_fill = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel_fill)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel_fill)

    # Cari kontur terbesar (telur)
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        # Fallback: kembalikan citra asli di-resize ke canvas
        canvas = np.zeros((target_canvas, target_canvas, 3), dtype=np.uint8)
        scale_fb = min(target_canvas / w_orig, target_canvas / h_orig) * 0.9
        nw_fb = int(w_orig * scale_fb)
        nh_fb = int(h_orig * scale_fb)
        resized_fb = cv2.resize(img_rgb, (nw_fb, nh_fb), interpolation=cv2.INTER_LANCZOS4)
        x_off_fb = (target_canvas - nw_fb) // 2
        y_off_fb = (target_canvas - nh_fb) // 2
        canvas[y_off_fb:y_off_fb + nh_fb, x_off_fb:x_off_fb + nw_fb] = resized_fb
        if return_transform:
            return canvas, {"norm_scale": scale_fb, "tx": float(x_off_fb), "ty": float(y_off_fb), "ellipse_orig": None}
        return canvas

    biggest = max(contours, key=cv2.contourArea)
    if len(biggest) < 5:
        # Tidak cukup poin untuk fitEllipse — fallback
        canvas = np.zeros((target_canvas, target_canvas, 3), dtype=np.uint8)
        scale_fb = min(target_canvas / w_orig, target_canvas / h_orig) * 0.9
        nw_fb = int(w_orig * scale_fb)
        nh_fb = int(h_orig * scale_fb)
        resized_fb = cv2.resize(img_rgb, (nw_fb, nh_fb), interpolation=cv2.INTER_LANCZOS4)
        x_off_fb = (target_canvas - nw_fb) // 2
        y_off_fb = (target_canvas - nh_fb) // 2
        canvas[y_off_fb:y_off_fb + nh_fb, x_off_fb:x_off_fb + nw_fb] = resized_fb
        if return_transform:
            return canvas, {"norm_scale": scale_fb, "tx": float(x_off_fb), "ty": float(y_off_fb), "ellipse_orig": None}
        return canvas

    # 2. fitEllipse pada kontur telur
    ellipse_raw = cv2.fitEllipse(biggest)
    (cx_raw, cy_raw), (ax1_raw, ax2_raw), angle_raw = ellipse_raw
    major_axis = max(ax1_raw, ax2_raw)

    if major_axis <= 0:
        major_axis = max(h_orig, w_orig) * 0.6

    # 3. Hitung skala dan translasi affine
    norm_scale = target_egg_height / major_axis
    center_canvas = target_canvas / 2.0

    # Translasi: setelah scaling, center telur harus di tengah canvas
    tx = center_canvas - cx_raw * norm_scale
    ty = center_canvas - cy_raw * norm_scale

    # 4. Affine warpAffine (scale + translate, tanpa rotasi)
    M = np.array([
        [norm_scale, 0.0, tx],
        [0.0, norm_scale, ty]
    ], dtype=np.float64)

    warped = cv2.warpAffine(
        img_rgb, M, (target_canvas, target_canvas),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )

    # 5. Anti-aliased feathering mask (supersampled 2x)
    ss = 2  # supersampling factor
    ss_size = target_canvas * ss
    mask_ss = np.zeros((ss_size, ss_size), dtype=np.uint8)

    # Transform ellipse ke canvas coordinate
    ellipse_canvas = (
        (float(cx_raw * norm_scale + tx), float(cy_raw * norm_scale + ty)),
        (float(ax1_raw * norm_scale), float(ax2_raw * norm_scale)),
        float(angle_raw)
    )

    # Supersampled ellipse
    ellipse_ss = (
        (ellipse_canvas[0][0] * ss, ellipse_canvas[0][1] * ss),
        (ellipse_canvas[1][0] * ss, ellipse_canvas[1][1] * ss),
        ellipse_canvas[2]
    )

    cv2.ellipse(mask_ss, (
        (int(ellipse_ss[0][0]), int(ellipse_ss[0][1])),
        (int(ellipse_ss[1][0]), int(ellipse_ss[1][1])),
        ellipse_ss[2]
    ), 255, -1)

    # Downsample mask untuk anti-aliasing
    mask_aa = cv2.resize(mask_ss, (target_canvas, target_canvas), interpolation=cv2.INTER_AREA)

    # Gaussian blur untuk feathering yang halus
    mask_feather = cv2.GaussianBlur(mask_aa, (7, 7), 0)
    mask_3ch = np.repeat(mask_feather[:, :, np.newaxis].astype(np.float32) / 255.0, 3, axis=2)

    # 6. Apply mask: background hitam pekat
    result = (warped.astype(np.float32) * mask_3ch).astype(np.uint8)

    # 7. Return
    if return_transform:
        ellipse_raw_tuple = (
            (float(cx_raw), float(cy_raw)),
            (float(ax1_raw), float(ax2_raw)),
            float(angle_raw)
        )
        transform_info = {
            "norm_scale": float(norm_scale),
            "tx": float(tx),
            "ty": float(ty),
            "ellipse_orig": ellipse_raw_tuple
        }
        return result, transform_info

    return result


def detect_physical_cracks(img_rgb: np.ndarray, ellipse_orig: Tuple) -> Tuple[bool, list]:
    """
    Verifikasi fisik fraktur retak cangkang berbasis Computer Vision (OpenCV):
    - Mengisolasi area cangkang telur bagian dalam (erosi adaptif 12% agar tepi elips dan bayangan meja terpotong).
    - Menerapkan multi-directional blackhat (vertikal 3x13, horizontal 13x3, dan omni 7x7) untuk menonjolkan garis retakan samar/rambut.
    - Directional closing (2x17 vertikal) untuk menjembatani segmen mikro retakan collinear tanpa menggabungkan derau samping.
    - Filter fraktur retak nyata: rasio aspek memanjang (aspect >= 2.0), panjang diagonal (diag >= 35.0px),
      fill_ratio < 0.65, serta berada di dalam badan cangkang telur (norm_dist <= 0.82).
    - Clustering spasial presisi: menggabungkan komponen fraktur berdekatan (dist <= 25px) dan membungkusnya dalam bounding box presisi.
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape[:2]

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(mask, ellipse_orig, 255, -1)
    erode_r = max(7, int(min(ellipse_orig[1]) * 0.12))
    kernel_e = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_r, erode_r))
    mask_inner = cv2.erode(mask, kernel_e)

    # Hitung rata-rata intensitas cangkang untuk adaptive threshold (retakan memiliki kontras tinggi)
    egg_mean = float(cv2.mean(gray, mask=mask_inner)[0])
    adaptive_thresh = max(30, int(egg_mean * 0.25))

    # Directional blackhat (vertikal & horizontal) khusus mendeteksi fraktur garis memanjang
    k_v = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 15))
    k_h = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    bh_v = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k_v)
    bh_h = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k_h)
    bh_max = np.maximum(bh_v, bh_h)

    bh_egg = cv2.bitwise_and(bh_max, bh_max, mask=mask_inner)
    _, crack_thresh = cv2.threshold(bh_egg, adaptive_thresh, 255, cv2.THRESH_BINARY)

    # Morfologi pembersihan: hapus bintik mikro dengan opening
    clean_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    crack_thresh = cv2.morphologyEx(crack_thresh, cv2.MORPH_OPEN, clean_kernel)

    # Directional bridge untuk menyambung fraktur rambut yang terputus pori
    bridge_v = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 15))
    crack_bridged = cv2.morphologyEx(crack_thresh, cv2.MORPH_CLOSE, bridge_v)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(crack_bridged)

    cracks = []
    (cx_e, cy_e), (ax1_e, ax2_e), _ = ellipse_orig
    rad_x = max(1.0, ax1_e / 2.0)
    rad_y = max(1.0, ax2_e / 2.0)

    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        cw = int(stats[i, cv2.CC_STAT_WIDTH])
        ch = int(stats[i, cv2.CC_STAT_HEIGHT])
        left = int(stats[i, cv2.CC_STAT_LEFT])
        top = int(stats[i, cv2.CC_STAT_TOP])
        diag = float(np.sqrt(cw**2 + ch**2))
        aspect = float(max(cw, ch) / max(1, min(cw, ch)))
        fill_ratio = float(area / max(1, cw * ch))
        thickness = min(cw, ch)

        cx_box = left + cw / 2.0
        cy_box = top + ch / 2.0
        norm_dist = (((cx_box - cx_e)**2) / (rad_x**2)) + (((cy_box - cy_e)**2) / (rad_y**2))

        # Kriteria fraktur fisik retak nyata:
        # 1. Aspek rasio memanjang garis (aspect >= 2.0)
        # 2. Panjang diagonal fraktur nyata (diag >= 35.0px)
        # 3. Ketebalan linier rambut (thickness <= 25px, bukan bayangan permukaan luas)
        # 4. Batas luas komponen garis (area <= 1500px, bukan noda/shading cangkang)
        # 5. Bukan bidang padat (fill_ratio < 0.65)
        # 6. Berada di dalam badan cangkang telur (norm_dist <= 0.82)
        if aspect >= 2.0 and diag >= 35.0 and thickness <= 25 and area <= 1500 and fill_ratio < 0.65 and norm_dist <= 0.82:
            cracks.append({
                "bbox": [left, top, left + cw, top + ch],
                "area": area,
                "aspect": round(aspect, 2),
                "diag": round(diag, 1),
                "cx": cx_box,
                "cy": cy_box
            })

    # Deduplikasi & clustering spasial komponen retakan berdekatan
    if len(cracks) > 1:
        merged = []
        used = set()
        for i, c1 in enumerate(cracks):
            if i in used:
                continue
            b1 = c1["bbox"]
            cluster = [c1]
            for j in range(i + 1, len(cracks)):
                if j in used:
                    continue
                b2 = cracks[j]["bbox"]
                dx = max(0, max(b1[0], b2[0]) - min(b1[2], b2[2]))
                dy = max(0, max(b1[1], b2[1]) - min(b1[3], b2[3]))
                dist = np.sqrt(dx**2 + dy**2)
                if dist <= 25:
                    cluster.append(cracks[j])
                    used.add(j)

            cl_x1 = max(0, min(c["bbox"][0] for c in cluster) - 8)
            cl_y1 = max(0, min(c["bbox"][1] for c in cluster) - 10)
            cl_x2 = min(w, max(c["bbox"][2] for c in cluster) + 8)
            cl_y2 = min(h, max(c["bbox"][3] for c in cluster) + 10)
            total_area = sum(c["area"] for c in cluster)
            cl_diag = np.sqrt((cl_x2 - cl_x1)**2 + (cl_y2 - cl_y1)**2)
            cl_asp = max(cl_x2 - cl_x1, cl_y2 - cl_y1) / max(1, min(cl_x2 - cl_x1, cl_y2 - cl_y1))
            merged.append({
                "area": total_area,
                "diagonal_length": round(float(cl_diag), 1),
                "aspect_ratio": round(float(cl_asp), 2),
                "bbox": [cl_x1, cl_y1, cl_x2, cl_y2]
            })
        cracks = merged
    elif len(cracks) == 1:
        b = cracks[0]["bbox"]
        cl_x1 = max(0, b[0] - 8)
        cl_y1 = max(0, b[1] - 10)
        cl_x2 = min(w, b[2] + 8)
        cl_y2 = min(h, b[3] + 10)
        cracks[0]["bbox"] = [cl_x1, cl_y1, cl_x2, cl_y2]

    return len(cracks) > 0, cracks

