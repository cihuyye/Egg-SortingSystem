from typing import Any, Dict, Tuple
import cv2
import numpy as np

def analyze_candling_quality(image_np: np.ndarray, px_to_mm: float = 0.15) -> Dict[str, Any]:
    """
    Modul Ekstraksi Fitur Candling Telur berdasarkan SNI 3926:2023 Tabel 1 (Poin 2 & 3).
    Didesain khusus untuk kamera samping (side-view) atau atas, serta adaptif terhadap
    orientasi telur di konveyor (telur tidur horizontal kiri/kanan, berdiri, maupun miring).
    Menerima citra RGB (Pillow / Matplotlib format).
    """
    hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    h_img, w_img = image_np.shape[:2]

    # 1. Segmentasi Telur Tembus Cahaya
    lower_egg = np.array([0, 20, 40])
    upper_egg = np.array([179, 255, 255])
    mask_egg_hsv = cv2.inRange(hsv, lower_egg, upper_egg)
    
    _, mask_egg_gray = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)
    mask_egg = cv2.bitwise_or(mask_egg_hsv, mask_egg_gray)
    
    kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask_egg = cv2.morphologyEx(mask_egg, cv2.MORPH_CLOSE, kernel_clean)
    mask_egg = cv2.morphologyEx(mask_egg, cv2.MORPH_OPEN, kernel_clean)

    contours_egg, _ = cv2.findContours(mask_egg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    luas_telur: float = 0.0
    tinggi_kantung_mm: float = 0.0
    luas_bercak: float = 0.0
    rata_kecerahan: float = 0.0
    dark_spot_ratio: float = 0.0
    orientation_desc: str = "Tidak Terdeteksi"
    blunt_pole: str = "top"

    if contours_egg:
        c_egg = max(contours_egg, key=cv2.contourArea)
        area_egg_val = cv2.contourArea(c_egg)
        
        if area_egg_val > (h_img * w_img * 0.03):
            luas_telur = float(area_egg_val)
            x, y, w, h = cv2.boundingRect(c_egg)
            
            egg_mask = np.zeros_like(mask_egg)
            cv2.drawContours(egg_mask, [c_egg], -1, 255, -1)
            rata_kecerahan = float(cv2.mean(hsv[:, :, 2], mask=egg_mask)[0])

            # 2. DETEKSI ORIENTASI TELUR
            M = cv2.moments(c_egg)
            cx = int(M["m10"] / M["m00"]) if M["m00"] != 0 else x + w // 2
            mid_x = x + w / 2.0
            mid_y = y + h / 2.0
            cy = int(M["m01"] / M["m00"]) if M["m00"] != 0 else y + h // 2

            air_roi_mask = np.zeros_like(mask_egg)

            if w >= h:
                # Telur tidur horizontal
                pole_w = int(w * 0.25)
                mask_left = np.zeros_like(mask_egg)
                cv2.rectangle(mask_left, (x, y), (x + pole_w, y + h), 255, -1)
                mean_left = cv2.mean(hsv[:, :, 2], mask=cv2.bitwise_and(egg_mask, mask_left))[0]

                mask_right = np.zeros_like(mask_egg)
                cv2.rectangle(mask_right, (x + w - pole_w, y), (x + w, y + h), 255, -1)
                mean_right = cv2.mean(hsv[:, :, 2], mask=cv2.bitwise_and(egg_mask, mask_right))[0]

                score_left = (mean_left * 0.6) + (1.0 if cx < mid_x else 0.0) * 40
                score_right = (mean_right * 0.6) + (1.0 if cx >= mid_x else 0.0) * 40

                if score_left >= score_right:
                    blunt_pole = "kiri"
                    orientation_desc = "Tidur Horizontal (Ujung Tumpul di Kiri)"
                    air_roi_mask = cv2.bitwise_and(egg_mask, mask_left)
                    measure_axis = "width"
                else:
                    blunt_pole = "kanan"
                    orientation_desc = "Tidur Horizontal (Ujung Tumpul di Kanan)"
                    air_roi_mask = cv2.bitwise_and(egg_mask, mask_right)
                    measure_axis = "width"
            else:
                # Telur berdiri vertikal
                pole_h = int(h * 0.25)
                mask_top = np.zeros_like(mask_egg)
                cv2.rectangle(mask_top, (x, y), (x + w, y + pole_h), 255, -1)
                mean_top = cv2.mean(hsv[:, :, 2], mask=cv2.bitwise_and(egg_mask, mask_top))[0]

                mask_bottom = np.zeros_like(mask_egg)
                cv2.rectangle(mask_bottom, (x, y + h - pole_h), (x + w, y + h), 255, -1)
                mean_bottom = cv2.mean(hsv[:, :, 2], mask=cv2.bitwise_and(egg_mask, mask_bottom))[0]

                score_top = (mean_top * 0.6) + (1.0 if cy < mid_y else 0.0) * 40
                score_bottom = (mean_bottom * 0.6) + (1.0 if cy >= mid_y else 0.0) * 40

                if score_top >= score_bottom:
                    blunt_pole = "atas"
                    orientation_desc = "Berdiri Vertikal (Ujung Tumpul di Atas)"
                    air_roi_mask = cv2.bitwise_and(egg_mask, mask_top)
                    measure_axis = "height"
                else:
                    blunt_pole = "bawah"
                    orientation_desc = "Berdiri Vertikal (Ujung Tumpul di Bawah)"
                    air_roi_mask = cv2.bitwise_and(egg_mask, mask_bottom)
                    measure_axis = "height"

            # 3. KANTUNG UDARA
            v_channel = hsv[:, :, 2]
            v_pole = cv2.bitwise_and(v_channel, v_channel, mask=air_roi_mask)
            non_zero_v = v_pole[air_roi_mask > 0]
            if len(non_zero_v) > 50:
                thresh_val = float(np.percentile(non_zero_v, 65))
                _, mask_air = cv2.threshold(v_pole, int(max(thresh_val, 110.0)), 255, cv2.THRESH_BINARY)
                mask_air = cv2.morphologyEx(mask_air, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
                
                cnts_air, _ = cv2.findContours(mask_air, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if cnts_air:
                    c_air_max = max(cnts_air, key=cv2.contourArea)
                    if cv2.contourArea(c_air_max) > 40:
                        _, _, wa, ha = cv2.boundingRect(c_air_max)
                        pole_depth_px = wa if measure_axis == "width" else ha
                        tinggi_kantung_mm = float(pole_depth_px * px_to_mm)

            # 4. CACAT BERCAK DARAH
            lower_dark = np.array([0, 0, 0])
            upper_dark = np.array([179, 255, 45])
            mask_dark = cv2.inRange(hsv, lower_dark, upper_dark)
            mask_bercak_raw = cv2.bitwise_and(mask_dark, mask_dark, mask=egg_mask)
            
            kernel_speckle = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask_bercak = cv2.morphologyEx(mask_bercak_raw, cv2.MORPH_OPEN, kernel_speckle)

            cnts_bercak, _ = cv2.findContours(mask_bercak, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cb in cnts_bercak:
                if cv2.contourArea(cb) > 120:
                    luas_bercak += float(cv2.contourArea(cb))

            if luas_telur > 0:
                dark_spot_ratio = luas_bercak / luas_telur

    # 5. STANDAR SNI 3926:2023
    tinggi_kantung_cm = tinggi_kantung_mm / 10.0
    
    if dark_spot_ratio > 0.03 or luas_bercak > 250:
        sni_candling_grade = "Mutu III (Reject)"
        reason = "Terdeteksi bercak darah / noda asing di dalam isi telur (SNI 3926:2023 Tabel 1 Poin 3)"
    elif tinggi_kantung_cm > 0.9 or (luas_telur > 0 and rata_kecerahan < 60):
        sni_candling_grade = "Mutu III (Reject)"
        reason = f"Kedalaman kantung udara > 0.9 cm ({tinggi_kantung_cm:.2f} cm) atau telur busuk gelap"
    elif 0.5 <= tinggi_kantung_cm <= 0.9:
        sni_candling_grade = "Mutu II"
        reason = f"Kedalaman kantung udara 0.5 - 0.9 cm ({tinggi_kantung_cm:.2f} cm), bebas bercak darah"
    else:
        sni_candling_grade = "Mutu I"
        reason = f"Kedalaman kantung udara < 0.5 cm ({tinggi_kantung_cm:.2f} cm), kondisi translusen bersih"

    return {
        "mean_brightness": round(rata_kecerahan, 2),
        "dark_spot_ratio": round(dark_spot_ratio, 4),
        "air_cell_depth_cm": round(tinggi_kantung_cm, 2),
        "air_cell_depth_mm": round(tinggi_kantung_mm, 2),
        "candling_status": sni_candling_grade,
        "reason": reason,
        "orientation_detected": orientation_desc,
        "blunt_pole_position": blunt_pole,
        "standard_reference": "SNI 3926:2023 Tabel 1 Poin 2 & Poin 3"
    }


def process_candling_frame(frame_bgr: np.ndarray, px_to_mm: float = 0.15) -> Tuple[Dict[str, Any], np.ndarray]:
    """
    Fungsi khusus untuk pemrosesan video streaming real-time (kamera BGR OpenCV).
    Menghasilkan hasil evaluasi SNI dan citra beranotasi visual lengkap.
    """
    visual_frame: np.ndarray = frame_bgr.copy()
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    h_img, w_img = frame_bgr.shape[:2]

    lower_egg = np.array([0, 20, 40])
    upper_egg = np.array([179, 255, 255])
    mask_egg_hsv = cv2.inRange(hsv, lower_egg, upper_egg)
    _, mask_egg_gray = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)
    mask_egg = cv2.bitwise_or(mask_egg_hsv, mask_egg_gray)
    
    kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask_egg = cv2.morphologyEx(mask_egg, cv2.MORPH_CLOSE, kernel_clean)
    mask_egg = cv2.morphologyEx(mask_egg, cv2.MORPH_OPEN, kernel_clean)

    contours_egg, _ = cv2.findContours(mask_egg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    luas_telur: float = 0.0
    tinggi_kantung_mm: float = 0.0
    luas_bercak: float = 0.0
    rata_kecerahan: float = 0.0
    dark_spot_ratio: float = 0.0
    orientation_desc: str = "Tidak Terdeteksi"
    blunt_pole: str = "top"

    if contours_egg:
        c_egg = max(contours_egg, key=cv2.contourArea)
        area_egg_val = cv2.contourArea(c_egg)
        
        if area_egg_val > (h_img * w_img * 0.03):
            luas_telur = float(area_egg_val)
            x, y, w, h = cv2.boundingRect(c_egg)
            
            egg_mask = np.zeros_like(mask_egg)
            cv2.drawContours(egg_mask, [c_egg], -1, 255, -1)
            cv2.drawContours(visual_frame, [c_egg], -1, (0, 255, 0), 2)

            rata_kecerahan = float(cv2.mean(hsv[:, :, 2], mask=egg_mask)[0])

            M = cv2.moments(c_egg)
            cx = int(M["m10"] / M["m00"]) if M["m00"] != 0 else x + w // 2
            cy = int(M["m01"] / M["m00"]) if M["m00"] != 0 else y + h // 2
            mid_x = x + w / 2.0
            mid_y = y + h / 2.0

            air_roi_mask = np.zeros_like(mask_egg)

            if w >= h:
                pole_w = int(w * 0.25)
                mask_left = np.zeros_like(mask_egg)
                cv2.rectangle(mask_left, (x, y), (x + pole_w, y + h), 255, -1)
                mean_left = cv2.mean(hsv[:, :, 2], mask=cv2.bitwise_and(egg_mask, mask_left))[0]

                mask_right = np.zeros_like(mask_egg)
                cv2.rectangle(mask_right, (x + w - pole_w, y), (x + w, y + h), 255, -1)
                mean_right = cv2.mean(hsv[:, :, 2], mask=cv2.bitwise_and(egg_mask, mask_right))[0]

                score_left = (mean_left * 0.6) + (1.0 if cx < mid_x else 0.0) * 40
                score_right = (mean_right * 0.6) + (1.0 if cx >= mid_x else 0.0) * 40

                if score_left >= score_right:
                    blunt_pole = "kiri"
                    orientation_desc = "Tidur Horizontal (Ujung Tumpul di Kiri)"
                    air_roi_mask = cv2.bitwise_and(egg_mask, mask_left)
                    cv2.rectangle(visual_frame, (x, y), (x + pole_w, y + h), (255, 255, 0), 2)
                    measure_axis = "width"
                else:
                    blunt_pole = "kanan"
                    orientation_desc = "Tidur Horizontal (Ujung Tumpul di Kanan)"
                    air_roi_mask = cv2.bitwise_and(egg_mask, mask_right)
                    cv2.rectangle(visual_frame, (x + w - pole_w, y), (x + w, y + h), (255, 255, 0), 2)
                    measure_axis = "width"
            else:
                pole_h = int(h * 0.25)
                mask_top = np.zeros_like(mask_egg)
                cv2.rectangle(mask_top, (x, y), (x + w, y + pole_h), 255, -1)
                mean_top = cv2.mean(hsv[:, :, 2], mask=cv2.bitwise_and(egg_mask, mask_top))[0]

                mask_bottom = np.zeros_like(mask_egg)
                cv2.rectangle(mask_bottom, (x, y + h - pole_h), (x + w, y + h), 255, -1)
                mean_bottom = cv2.mean(hsv[:, :, 2], mask=cv2.bitwise_and(egg_mask, mask_bottom))[0]

                score_top = (mean_top * 0.6) + (1.0 if cy < mid_y else 0.0) * 40
                score_bottom = (mean_bottom * 0.6) + (1.0 if cy >= mid_y else 0.0) * 40

                if score_top >= score_bottom:
                    blunt_pole = "atas"
                    orientation_desc = "Berdiri Vertikal (Ujung Tumpul di Atas)"
                    air_roi_mask = cv2.bitwise_and(egg_mask, mask_top)
                    cv2.rectangle(visual_frame, (x, y), (x + w, y + pole_h), (255, 255, 0), 2)
                    measure_axis = "height"
                else:
                    blunt_pole = "bawah"
                    orientation_desc = "Berdiri Vertikal (Ujung Tumpul di Bawah)"
                    air_roi_mask = cv2.bitwise_and(egg_mask, mask_bottom)
                    cv2.rectangle(visual_frame, (x, y + h - pole_h), (x + w, y + h), (255, 255, 0), 2)
                    measure_axis = "height"

            v_channel = hsv[:, :, 2]
            v_pole = cv2.bitwise_and(v_channel, v_channel, mask=air_roi_mask)
            non_zero_v = v_pole[air_roi_mask > 0]
            if len(non_zero_v) > 50:
                thresh_val = float(np.percentile(non_zero_v, 65))
                _, mask_air = cv2.threshold(v_pole, int(max(thresh_val, 110.0)), 255, cv2.THRESH_BINARY)
                mask_air = cv2.morphologyEx(mask_air, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
                
                cnts_air, _ = cv2.findContours(mask_air, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if cnts_air:
                    c_air_max = max(cnts_air, key=cv2.contourArea)
                    if cv2.contourArea(c_air_max) > 40:
                        _, _, wa, ha = cv2.boundingRect(c_air_max)
                        pole_depth_px = wa if measure_axis == "width" else ha
                        tinggi_kantung_mm = float(pole_depth_px * px_to_mm)

            lower_dark = np.array([0, 0, 0])
            upper_dark = np.array([179, 255, 45])
            mask_dark = cv2.inRange(hsv, lower_dark, upper_dark)
            mask_bercak_raw = cv2.bitwise_and(mask_dark, mask_dark, mask=egg_mask)
            
            kernel_speckle = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask_bercak = cv2.morphologyEx(mask_bercak_raw, cv2.MORPH_OPEN, kernel_speckle)

            cnts_bercak, _ = cv2.findContours(mask_bercak, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cb in cnts_bercak:
                if cv2.contourArea(cb) > 120:
                    luas_bercak += float(cv2.contourArea(cb))
                    xb, yb, wb, hb = cv2.boundingRect(cb)
                    cv2.rectangle(visual_frame, (xb, yb), (xb + wb, yb + hb), (0, 0, 255), 2)
                    cv2.putText(visual_frame, "CACAT/BERCAK", (xb, max(yb - 5, 15)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

            if luas_telur > 0:
                dark_spot_ratio = luas_bercak / luas_telur

    tinggi_kantung_cm = tinggi_kantung_mm / 10.0
    
    if dark_spot_ratio > 0.03 or luas_bercak > 250:
        sni_candling_grade = "Mutu III (Reject)"
        reason = "Terdeteksi bercak darah / noda asing (SNI 3926:2023)"
    elif tinggi_kantung_cm > 0.9 or (luas_telur > 0 and rata_kecerahan < 60):
        sni_candling_grade = "Mutu III (Reject)"
        reason = f"Kantung udara > 0.9 cm ({tinggi_kantung_cm:.2f} cm) / busuk gelap"
    elif 0.5 <= tinggi_kantung_cm <= 0.9:
        sni_candling_grade = "Mutu II"
        reason = f"Kantung udara 0.5 - 0.9 cm ({tinggi_kantung_cm:.2f} cm)"
    else:
        sni_candling_grade = "Mutu I"
        reason = f"Kantung udara < 0.5 cm ({tinggi_kantung_cm:.2f} cm)"

    result_dict: Dict[str, Any] = {
        "mean_brightness": round(rata_kecerahan, 2),
        "dark_spot_ratio": round(dark_spot_ratio, 4),
        "air_cell_depth_cm": round(tinggi_kantung_cm, 2),
        "air_cell_depth_mm": round(tinggi_kantung_mm, 2),
        "candling_status": sni_candling_grade,
        "reason": reason,
        "orientation_detected": orientation_desc,
        "blunt_pole_position": blunt_pole,
        "standard_reference": "SNI 3926:2023 Tabel 1 Poin 2 & Poin 3"
    }

    return result_dict, visual_frame
