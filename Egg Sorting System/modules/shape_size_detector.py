from typing import Any, Dict, Optional
import cv2
import numpy as np

def measure_egg_shape_and_size(image_np: np.ndarray, base_px_to_mm: float = 0.135) -> Optional[Dict[str, Any]]:
    """
    Mengukur dimensi panjang, lebar, dan Egg Shape Index (ESI/SI) telur
    berdasarkan standar SNI 3926:2023 (Tabel 1 & Pasal 4.2).
    Dilengkapi normalisasi skala dinamis terhadap resolusi citra kamera agar
    estimasi dimensi dan bobot tetap akurat di berbagai resolusi foto.
    """
    h_orig, w_orig = image_np.shape[:2]
    if h_orig == 0 or w_orig == 0:
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
    
    white_pixel_ratio = np.sum(thresh == 255) / (new_h * new_w)
    if white_pixel_ratio > 0.5:
        thresh = cv2.bitwise_not(thresh)
        
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
        
    valid_contours = [c for c in contours if (0.05 * new_h * new_w) < cv2.contourArea(c) < (0.95 * new_h * new_w)]
    best_cnt = max(valid_contours, key=cv2.contourArea) if valid_contours else max(contours, key=cv2.contourArea)
        
    if len(best_cnt) < 5:
        return None
        
    ellipse = cv2.fitEllipse(best_cnt)
    (cx, cy), (ax1, ax2), angle = ellipse
    
    minor_axis = min(ax1, ax2) # Lebar (W) dalam skala 640px
    major_axis = max(ax1, ax2) # Panjang / Tinggi (L) dalam skala 640px
    
    length_mm = major_axis * base_px_to_mm
    width_mm = minor_axis * base_px_to_mm
    
    # Rumus SNI 3926:2023 Lampiran A.1.1.3.1: SI = (W / L) * 100
    si = (width_mm / length_mm) * 100.0 if length_mm > 0 else 0
    
    # Standar SNI 3926:2023 Tabel 1: Normal = SI 72 sampai 76.
    # Di atas 76 = cenderung bulat, di bawah 72 = cenderung lonjong
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
    
    # Klasifikasi Bobot SNI 3926:2023 Pasal 4.2
    # Kecil (< 50 g), Sedang (50 - 60 g), Besar (> 60 g), Ekstra Besar (> 65g)
    if weight_gram < 50.0:
        size_category = "Kecil (< 50g)"
    elif 50.0 <= weight_gram <= 60.0:
        size_category = "Sedang (50g - 60g)"
    elif 60.0 < weight_gram <= 65.0:
        size_category = "Besar (60g - 65g)"
    else:
        size_category = "Ekstra Besar (> 65g)"
        
    return {
        "length_mm": round(length_mm, 2),
        "width_mm": round(width_mm, 2),
        "esi_percentage": round(si, 2),
        "shape_status": shape_status,
        "estimated_weight_gram": round(weight_gram, 2),
        "size_category": size_category,
        "standard_reference": "SNI 3926:2023 Tabel 1 & Pasal 4.2"
    }
