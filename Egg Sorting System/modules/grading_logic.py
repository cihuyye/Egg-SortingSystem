from typing import Any, Dict

def evaluate_final_grade(
    yolo_defect_class: str,
    yolo_confidence: float,
    shape_dict: Dict[str, Any],
    candling_dict: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Logika Pengambilan Keputusan Akhir Mutu Telur Berdasarkan SNI 3926:2023.
    Mengintegrasikan langsung hasil klasifikasi YOLOv8 (keutuhan & kebersihan cangkang)
    dengan pengukuran geometri dan candling OpenCV tanpa ketergantungan model luar (SVM).

    Parameter Standar SNI 3926:2023:
    - Mutu I (Grade A):
      * Cangkang utuh & bersih (YOLOv8: 'good')
      * Bentuk normal (Shape Index 72 - 76)
      * Kedalaman kantung udara < 0.5 cm
      * Bebas bercak darah / noda asing
    - Mutu II (Grade B):
      * Cangkang utuh, boleh sedikit noda (YOLOv8: 'dirt' atau 'good')
      * Bentuk boleh sedikit di luar batas normal
      * Kedalaman kantung udara 0.5 - 0.9 cm
      * Bebas bercak darah
    - Mutu III (Reject):
      * Cangkang retak (YOLOv8: 'cracked')
      * Kedalaman kantung udara > 0.9 cm atau telur busuk gelap
      * Terdeteksi bercak darah / noda asing di dalam isi telur
    """
    si = float(shape_dict.get("esi_percentage", 74.0))
    size_cat = str(shape_dict.get("size_category", "Sedang"))
    weight_g = float(shape_dict.get("estimated_weight_gram", 55.0))
    
    air_depth_cm = float(candling_dict.get("air_cell_depth_cm", 0.3))
    dark_ratio = float(candling_dict.get("dark_spot_ratio", 0.0))
    brightness = float(candling_dict.get("mean_brightness", 180.0))
    candling_status = str(candling_dict.get("candling_status", "Mutu I"))

    # ATURAN 1: EVALUASI MUTU III (REJECT)
    # Kondisi retak cangkang oleh YOLOv8 adalah kondisi reject mutlak
    if yolo_defect_class == "cracked":
        return {
            "final_grade": "Mutu III (Reject)",
            "grade_code": "MUTU_III",
            "yolo_primary_factor": f"Retak cangkang terdeteksi ({yolo_confidence:.1%})",
            "sorting_action": "Alihkan ke wadah reject (jangan sampai pecah di jalur konveyor)",
            "reason": f"Cangkang telur retak (cracked) dengan keyakinan YOLOv8 {yolo_confidence:.1%}",
            "standard_reference": "SNI 3926:2023 Tabel 1 (Kerabang: Retak)",
            "size_info": f"{size_cat} ({weight_g:.1f}g)"
        }

    # Kondisi kerusakan interior (candling)
    if "Mutu III" in candling_status or air_depth_cm > 0.9 or dark_ratio > 0.03 or brightness < 60.0:
        alasan_interior = "Terdeteksi bercak darah / noda asing" if dark_ratio > 0.03 else (
            f"Kantung udara > 0.9 cm ({air_depth_cm:.2f} cm)" if air_depth_cm > 0.9 else "Kondisi interior gelap/busuk"
        )
        return {
            "final_grade": "Mutu III (Reject)",
            "grade_code": "MUTU_III",
            "yolo_primary_factor": "Cangkang utuh, namun interior reject",
            "sorting_action": "Alihkan ke wadah reject",
            "reason": f"{alasan_interior} (SNI 3926:2023 Tabel 1 Poin 2/3)",
            "standard_reference": "SNI 3926:2023 Tabel 1 (Kantung Udara > 0.9 cm / Noda Asing)",
            "size_info": f"{size_cat} ({weight_g:.1f}g)"
        }

    # ATURAN 2: EVALUASI MUTU II (GRADE B)
    # Cangkang kotor (dirt), atau kantung udara 0.5-0.9 cm, atau Shape Index abnormal (<72 atau >76)
    is_dirty = (yolo_defect_class == "dirt")
    is_air_mutu_2 = (0.5 <= air_depth_cm <= 0.9)
    is_abnormal_shape = (si < 72.0 or si > 76.0)

    if is_dirty or is_air_mutu_2 or is_abnormal_shape:
        reasons = []
        if is_dirty:
            reasons.append(f"Cangkang terdapat noda kotoran (YOLOv8: dirt {yolo_confidence:.1%})")
        if is_air_mutu_2:
            reasons.append(f"Kedalaman kantung udara {air_depth_cm:.2f} cm (toleransi 0.5 - 0.9 cm)")
        if is_abnormal_shape:
            reasons.append(f"Indeks bentuk {si:.1f}% di luar standar normal SNI 72-76")

        action = "Alihkan ke jalur pencucian/pembersihan telur" if is_dirty else "Kemas ke tray kelas komersial / industri kue"
        
        return {
            "final_grade": "Mutu II (Grade B)",
            "grade_code": "MUTU_II",
            "yolo_primary_factor": f"Cangkang: {yolo_defect_class} ({yolo_confidence:.1%})",
            "sorting_action": action,
            "reason": "; ".join(reasons),
            "standard_reference": "SNI 3926:2023 Tabel 1 Syarat Mutu II",
            "size_info": f"{size_cat} ({weight_g:.1f}g)"
        }

    # ATURAN 3: EVALUASI MUTU I (GRADE A - SUPERIOR)
    # Lolos seluruh parameter ideal SNI 3926:2023
    return {
        "final_grade": "Mutu I (Grade A)",
        "grade_code": "MUTU_I",
        "yolo_primary_factor": f"Cangkang utuh & mulus bersih (YOLOv8: good {yolo_confidence:.1%})",
        "sorting_action": f"Kemas ke wadah ritel premium ({size_cat})",
        "reason": f"Cangkang utuh bersih, Shape Index ideal ({si:.1f}%), kantung udara segar (< 0.5 cm, aktual {air_depth_cm:.2f} cm), bebas bercak",
        "standard_reference": "SNI 3926:2023 Tabel 1 Syarat Mutu I",
        "size_info": f"{size_cat} ({weight_g:.1f}g)"
    }
