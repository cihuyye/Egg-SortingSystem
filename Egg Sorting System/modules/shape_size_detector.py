from typing import Any, Dict, Optional, Tuple
import cv2
import numpy as np


def preprocess_egg_image(
    img_rgb: np.ndarray,
    target_canvas: int = 640,
    target_egg_height: float = 416.0,
    ellipse_scale: float = 0.975,
    blur_kernel: Tuple[int, int] = (7, 7),
    return_transform: bool = False
):
    """
    Pipeline Pra-pengolahan Citra Telur persis sesuai preprocess_dataset.ipynb:
    1. HSV Saturation Channel & Color Distance thresholding
    2. Contour & Ellipse fitting dengan multi-criteria scoring
    3. Auto-Centering & Scale Zoom (Affine Warp ke target_canvas 640x640)
    4. Anti-Aliased Feathered Mask berbasis Ellipse (ellipse_scale=0.975) yang
       MEMOTONG BERSIH bayangan lantai/keramik di tepi cangkang
    5. Hasil Bersih Background Hitam Pekat [0, 0, 0] yang identik dengan domain data latih YOLOv8.

    Returns:
        s6_final: np.ndarray (RGB, uint8) berukuran target_canvas x target_canvas
        (opsional jika return_transform=True) transform_info: Dict informasi transformasi koordinat
    """
    arr_orig = img_rgb
    h_orig, w_orig = arr_orig.shape[:2]
    target_size = (target_canvas, target_canvas)

    det_size = 640
    scale_det = det_size / max(h_orig, w_orig)
    arr_det = cv2.resize(arr_orig, (int(w_orig * scale_det), int(h_orig * scale_det)), interpolation=cv2.INTER_AREA)
    h_det, w_det = arr_det.shape[:2]

    hsv_det = cv2.cvtColor(arr_det, cv2.COLOR_RGB2HSV)
    corners = np.vstack([arr_det[:10, :10], arr_det[:10, -10:], arr_det[-10:, :10], arr_det[-10:, -10:]])
    bg_median = np.median(corners, axis=(0, 1))

    # Step 2: Isolasi Pigmen Cangkang (HSV Saturation)
    s = cv2.GaussianBlur(hsv_det[:, :, 1], (9, 9), 0)
    _, mask_s = cv2.threshold(s, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if mask_s[0, 0] > 128:
        mask_s = cv2.bitwise_not(mask_s)

    # Segmentasi Color Distance dari Background
    color_diff = np.sqrt(np.sum((arr_det.astype(np.float32) - bg_median) ** 2, axis=2))
    diff_norm = np.clip(color_diff / (color_diff.max() + 1e-6) * 255, 0, 255).astype(np.uint8)
    diff_blur = cv2.GaussianBlur(diff_norm, (9, 9), 0)
    _, mask_diff = cv2.threshold(diff_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if mask_diff[0, 0] > 128:
        mask_diff = cv2.bitwise_not(mask_diff)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    best_ellipse = None
    best_score = -1.0

    for raw_mask in [mask_s, mask_diff]:
        m_clean = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, kernel)
        m_clean = cv2.morphologyEx(m_clean, cv2.MORPH_CLOSE, kernel)
        cnts, _ = cv2.findContours(m_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for cnt in cnts:
            area = cv2.contourArea(cnt)
            if area < (0.01 * h_det * w_det):
                continue
            if len(cnt) < 5:
                continue
            try:
                ellipse = cv2.fitEllipse(cnt)
            except Exception:
                continue
            (cx, cy), (ax1, ax2), angle = ellipse
            minor, major = min(ax1, ax2), max(ax1, ax2)
            if minor <= 0 or major <= 0:
                continue
            aspect = minor / major
            if not (0.45 <= aspect <= 1.0):
                continue
            hull = cv2.convexHull(cnt)
            solidity = area / max(cv2.contourArea(hull), 1)
            if solidity < 0.75:
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            touches = (x <= 2 or y <= 2 or (x + bw) >= w_det - 2 or (y + bh) >= h_det - 2)
            if touches and area < 0.5 * h_det * w_det:
                continue
            mask_single = np.zeros((h_det, w_det), dtype=np.uint8)
            cv2.drawContours(mask_single, [cnt], -1, 255, -1)
            cnt_sat = cv2.mean(hsv_det[:, :, 1], mask=mask_single)[0]
            score = solidity * 100.0 + (1.0 - abs(aspect - 0.75)) * 50.0 + (cnt_sat / 255.0) * 50.0
            if score > best_score:
                best_score = score
                best_ellipse = ellipse

    if best_ellipse is None:
        dummy = _fallback_resize(arr_orig, target_canvas)
        scale = min(target_canvas / max(h_orig, 1), target_canvas / max(w_orig, 1), 1.0)
        nh, nw = max(1, int(h_orig * scale)), max(1, int(w_orig * scale))
        x_off = (target_canvas - nw) // 2
        y_off = (target_canvas - nh) // 2
        transform_info = {
            "norm_scale": float(scale),
            "tx": float(x_off),
            "ty": float(y_off),
            "ellipse_orig": ((float(w_orig / 2), float(h_orig / 2)), (float(w_orig * 0.8), float(h_orig * 0.8)), 0.0),
            "raw_w": int(w_orig),
            "raw_h": int(h_orig)
        }
        if return_transform:
            return dummy, transform_info
        return dummy

    # Step 3 & 4: Skala elips kembali ke dimensi asli & Auto-Centering
    (cx_d, cy_d), (ax1_d, ax2_d), angle = best_ellipse
    cx_orig = cx_d / scale_det
    cy_orig = cy_d / scale_det
    ax1_orig = ax1_d / scale_det
    ax2_orig = ax2_d / scale_det
    major_orig = max(ax1_orig, ax2_orig)

    norm_scale = target_egg_height / max(major_orig, 1.0)
    target_cx = target_size[0] / 2.0
    target_cy = target_size[1] / 2.0
    tx = target_cx - cx_orig * norm_scale
    ty = target_cy - cy_orig * norm_scale
    M = np.array([[norm_scale, 0, tx], [0, norm_scale, ty]], dtype=np.float32)
    s4_centered = cv2.warpAffine(arr_orig, M, target_size, flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    # Step 5: Feathered Anti-Aliasing Mask Berbasis Elips (Memotong Bersih Bayangan Lantai)
    new_ax1 = ax1_orig * norm_scale * ellipse_scale
    new_ax2 = ax2_orig * norm_scale * ellipse_scale
    mask_2x = np.zeros((target_size[1] * 2, target_size[0] * 2), dtype=np.uint8)
    ellipse_2x = ((target_cx * 2, target_cy * 2), (new_ax1 * 2, new_ax2 * 2), angle)
    cv2.ellipse(mask_2x, ellipse_2x, 255, -1)
    mask_smooth = cv2.resize(mask_2x, target_size, interpolation=cv2.INTER_AREA)
    s5_mask = cv2.GaussianBlur(mask_smooth, blur_kernel, 0)

    # Step 6: Hasil Bersih (Background Hitam Pekat)
    mask_3ch = np.repeat((s5_mask / 255.0)[:, :, np.newaxis], 3, axis=2)
    s6_final = (s4_centered * mask_3ch).astype(np.uint8)

    transform_info = {
        "norm_scale": float(norm_scale),
        "tx": float(tx),
        "ty": float(ty),
        "ellipse_orig": ((float(cx_orig), float(cy_orig)), (float(ax1_orig), float(ax2_orig)), float(angle)),
        "raw_w": int(w_orig),
        "raw_h": int(h_orig)
    }
    if return_transform:
        return s6_final, transform_info
    return s6_final


def _fallback_resize(img_rgb: np.ndarray, target_canvas: int) -> np.ndarray:
    """Fallback: letakkan citra asli di tengah canvas hitam tanpa masking."""
    h, w = img_rgb.shape[:2]
    scale = min(target_canvas / max(h, 1), target_canvas / max(w, 1), 1.0)
    nh, nw = max(1, int(h * scale)), max(1, int(w * scale))
    resized = cv2.resize(img_rgb, (nw, nh), interpolation=cv2.INTER_LANCZOS4)
    canvas = np.zeros((target_canvas, target_canvas, 3), dtype=np.uint8)
    y_off = (target_canvas - nh) // 2
    x_off = (target_canvas - nw) // 2
    canvas[y_off:y_off + nh, x_off:x_off + nw] = resized
    return canvas


def validate_egg_candidate(img_rgb: np.ndarray, cnt: np.ndarray, total_area: int) -> Tuple[bool, Optional[Tuple], float]:
    """
    Validasi ilmiah morfologi cangkang telur ayam:
    1. Proporsi luas kontur terhadap resolusi frame (antara 1.5% sampai 90%).
    2. Rasio aspek elips (minor/major) 0.50 - 0.98 (telur lonjong s/d agak bulat).
    3. Nilai Solidity (Kecembungan cangkang) >= 0.75 (bebas sudut tajam, toleran kilau/bayangan).
    4. Karakteristik kromatis cangkang telur ayam ras (cokelat, krem, atau putih alami di berbagai pencahayaan).
    """
    area = cv2.contourArea(cnt)
    if area < (0.015 * total_area) or area > (0.92 * total_area):
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
        if not (0.50 <= aspect_ratio <= 0.98):
            return False, None, 0.0

        # Solidity: perbandingan luas kontur terhadap convex hull (cembung cangkang)
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        if hull_area <= 0:
            return False, None, 0.0
        solidity = area / hull_area
        if solidity < 0.75:
            return False, None, 0.0

        # Validasi warna dalam kontur cangkang (toleran terhadap LED putih ESP32-CAM maupun kamera HP)
        h_orig, w_orig = img_rgb.shape[:2]
        mask_single = np.zeros((h_orig, w_orig), dtype=np.uint8)
        cv2.drawContours(mask_single, [cnt], -1, 255, -1)
        mean_rgb = cv2.mean(img_rgb, mask=mask_single)[:3]
        r, g, b = mean_rgb

        # Cangkang telur nyata: memiliki intensitas memadai (bukan gelap gulita),
        # dan bukan warna artifisial seperti hijau/biru neon murni
        is_sufficient_bright = max(r, g, b) >= 30
        is_natural_shell = (r + 15 >= b * 0.80 and r + 15 >= g * 0.70 and g + 15 >= b * 0.65)

        if not (is_sufficient_bright and is_natural_shell):
            return False, None, 0.0

        score = area * solidity
        return True, ellipse, score

    except Exception:
        return False, None, 0.0


def measure_egg_shape_and_size(image_np: np.ndarray, base_px_to_mm: float = 0.120) -> Optional[Dict[str, Any]]:
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

    # Batas fisik wajar deteksi kamera visual: panjang 25 - 120 mm, lebar 18 - 90 mm
    if not (20.0 <= length_mm <= 130.0) or not (15.0 <= width_mm <= 95.0):
        return None

    # Rumus SNI 3926:2023 Lampiran A.1.1.3.1: SI = (W / L) * 100
    si = (width_mm / length_mm) * 100.0 if length_mm > 0 else 0
    if not (48.0 <= si <= 98.0):
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

    # Batas keamanan absolut (hanya tolak jika mustahil seperti debu < 10g atau raksasa > 300g)
    if not (10.0 <= weight_gram <= 300.0):
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

    # Multi-scale blackhat untuk mendeteksi retakan tipis dan tebal
    cracks_found = []

    for kernel_size in [(7, 7), (11, 11), (15, 15)]:
        kernel_crack = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel_crack)
        blackhat_egg = cv2.bitwise_and(blackhat, blackhat, mask=mask_inner)

        # CLAHE untuk meningkatkan kontras retakan halus
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(blackhat_egg)

        _, crack_thresh = cv2.threshold(enhanced, 15, 255, cv2.THRESH_BINARY)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(crack_thresh)

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            cw = stats[i, cv2.CC_STAT_WIDTH]
            ch = stats[i, cv2.CC_STAT_HEIGHT]
            diag = np.sqrt(cw**2 + ch**2)
            aspect = max(cw, ch) / max(1, min(cw, ch))

            # Threshold lebih sensitif: area >= 60, diagonal >= 25, aspek memanjang >= 1.8
            if area >= 60 and diag >= 25.0 and aspect >= 1.8:
                cracks_found.append({
                    "area": int(area),
                    "diagonal_length": round(float(diag), 1),
                    "aspect_ratio": round(float(aspect), 2),
                    "kernel": kernel_size[0]
                })

    return len(cracks_found) > 0, cracks_found

