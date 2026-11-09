from typing import Any, Dict, Optional, Tuple
import cv2
import numpy as np

def validate_egg_candidate(img_rgb: np.ndarray, cnt: np.ndarray, total_area: int) -> Tuple[bool, Optional[Tuple], float]:
    """
    Validasi ilmiah morfologi cangkang telur ayam:
    1. Proporsi luas kontur terhadap resolusi frame (antara 2.5% sampai 85%).
    2. Rasio aspek elips (minor/major) 0.55 - 0.95 (tidak pipih memanjang atau garis).
    3. Nilai Solidity (Kecembungan cangkang) >= 0.85 (bebas lekukan, siku, sudut tajam).
    4. Karakteristik kromatis cangkang telur ayam ras (cokelat/krem atau putih alami).
    """
    area = cv2.contourArea(cnt)
    if area < (0.025 * total_area) or area > (0.85 * total_area):
        return False, None, 0.0

    if len(cnt) < 5:
        return False, None, 0.0

    try:
        ellipse = cv2.fitEllipse(cnt)
        (cx, cy), (ax1, ax2), angle = ellipse
        minor = min(ax1, ax2)
        major = max(ax1, ax2)

        if minor <= 0 or major <= 0:
            return False, None, 0.0

        aspect_ratio = minor / major
        if not (0.55 <= aspect_ratio <= 0.95):
            return False, None, 0.0

        # Solidity: perbandingan luas kontur terhadap convex hull
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        if hull_area <= 0:
            return False, None, 0.0
        solidity = area / hull_area
        if solidity < 0.85:
            return False, None, 0.0

        # Validasi warna dalam kontur cangkang
        h_orig, w_orig = img_rgb.shape[:2]
        mask_single = np.zeros((h_orig, w_orig), dtype=np.uint8)
        cv2.drawContours(mask_single, [cnt], -1, 255, -1)
        mean_rgb = cv2.mean(img_rgb, mask=mask_single)[:3]
        r, g, b = mean_rgb

        is_warm_brown = (r > b * 1.04 and r > g * 0.88 and r > 60)
        is_white_shell = (r > 120 and g > 120 and b > 110 and abs(r - b) < 45)

        if not (is_warm_brown or is_white_shell):
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


def detect_physical_cracks(img_rgb: np.ndarray, ellipse_orig: Tuple) -> Tuple[bool, list]:
    """
    Verifikasi fisik fraktur retak cangkang berbasis Computer Vision (OpenCV):
    - Mengisolasi area cangkang telur bagian dalam (erosi 8% agar tepi elips & bayangan tidak terhitung).
    - Menerapkan transformasi blackhat morfologis untuk menonjolkan garis retakan gelap tipis.
    - Memfilter noise/pori-pori alami cangkang (pori-pori luas < 80px & diagonal < 35px,
      sedangkan retakan nyata memiliki luas >= 120px, diagonal >= 40px, dan rasio aspek memanjang >= 2.0).
    - Mengembalikan (has_crack: bool, crack_components: list).
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape[:2]

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(mask, ellipse_orig, 255, -1)
    erode_r = max(5, int(min(h, w) * 0.08))
    kernel_e = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_r, erode_r))
    mask_inner = cv2.erode(mask, kernel_e)

    kernel_crack = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel_crack)
    blackhat_egg = cv2.bitwise_and(blackhat, blackhat, mask=mask_inner)

    _, crack_thresh = cv2.threshold(blackhat_egg, 18, 255, cv2.THRESH_BINARY)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(crack_thresh)

    cracks_found = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        cw = stats[i, cv2.CC_STAT_WIDTH]
        ch = stats[i, cv2.CC_STAT_HEIGHT]
        diag = np.sqrt(cw**2 + ch**2)
        aspect = max(cw, ch) / max(1, min(cw, ch))

        if area >= 120 and diag >= 40.0 and aspect >= 2.0:
            cracks_found.append({
                "area": int(area),
                "diagonal_length": round(float(diag), 1),
                "aspect_ratio": round(float(aspect), 2)
            })

    return len(cracks_found) > 0, cracks_found

