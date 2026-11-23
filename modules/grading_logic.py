"""
================================================================================
LOGIKA PENGAMBILAN KEPUTUSAN MUTU TELUR (SNI 3926:2023)
Acuan Standar: Standar Nasional Indonesia (SNI 3926:2023)
Parameter Evaluasi:
- Kondisi Fisik Kerabang (Keutuhan & Kebersihan via YOLOv8-seg)
- Geometri Dimensi & Egg Shape Index (ESI % via OpenCV)
- Estimasi Bobot Telur & Kategori Ukuran (Tabel 2 SNI 3926:2023)
================================================================================
"""

from typing import Any, Dict, Optional


def evaluate_final_grade(
    yolo_defect_class: str,
    yolo_confidence: float,
    shape_dict: Dict[str, Any],
    dirt_percentage: float = 0.0,
    **kwargs: Any
) -> Dict[str, Any]:
    """
    Logika Pengambilan Keputusan Akhir Mutu Telur Berdasarkan SNI 3926:2023.
    Mengintegrasikan hasil klasifikasi cacat permukaan cangkang (YOLOv8-seg)
    dengan pengukuran geometri bentuk dan estimasi bobot telur (OpenCV).

    Parameter Standar SNI 3926:2023 (Tabel 1: Kerabang & Bentuk, Tabel 2: Bobot):
    - Mutu I (Grade A):
      * Kerabang utuh, halus, dan bersih (YOLOv8: 'good')
      * Bentuk normal (Egg Shape Index 72% - 76%)
    - Mutu II (Grade B):
      * Kerabang utuh, boleh sedikit bernoda kotoran (luas noda <= 1/32 permukaan, YOLOv8: 'dirt')
      * ATAU Bentuk boleh sedikit di luar batas normal ideal (< 72% atau > 76%)
    - Mutu III (Reject):
      * Kerabang retak / tidak utuh (YOLOv8: 'cracked')
      * ATAU Noda kotoran kerabang melebihi toleransi maksimal (> 1/16 atau > 6.25%)
    """
    si = float(shape_dict.get("esi_percentage", 74.0))
    size_cat = str(shape_dict.get("size_category", "Sedang"))
    weight_g = float(shape_dict.get("estimated_weight_gram", 55.0))

    # ==========================================================================
    # ATURAN 1: EVALUASI MUTU III (REJECT / AFKIR)
    # ==========================================================================
    # 1. Retak Cangkang (Cracked): Cacat mutu kritis, dilarang diedarkan
    if yolo_defect_class == "cracked":
        return {
            "final_grade": "Mutu III (Reject)",
            "grade_code": "MUTU_III",
            "yolo_primary_factor": f"Retak kerabang terdeteksi ({yolo_confidence:.1%})",
            "sorting_action": "Reject Ejector: Alihkan ke wadah karantina afkir (cegah kontaminasi)",
            "sni_action": "Dilarang diedarkan sebagai telur segar utuh (Tabel 1 SNI 3926:2023). Cacat kerabang retak rawan cemaran Salmonella spp. Alihkan ke karantina afkir/pemusnahan.",
            "reason": f"Kerabang telur retak (cracked) melanggar syarat keutuhan SNI 3926:2023 Tabel 1 (keyakinan YOLOv8: {yolo_confidence:.1%})",
            "standard_reference": "SNI 3926:2023 Tabel 1 (Kerabang: Tidak Utuh / Retak)",
            "size_info": f"{size_cat} ({weight_g:.1f}g)"
        }

    # 2. Noda Kotoran Parah: Melebihi batas toleransi maksimal SNI (> 1/16 atau > 6.25%)
    if dirt_percentage > 6.25:
        return {
            "final_grade": "Mutu III (Reject)",
            "grade_code": "MUTU_III",
            "yolo_primary_factor": f"Noda kotoran kerabang {dirt_percentage:.1f}% (> 6.25% batas SNI)",
            "sorting_action": "Reject Ejector: Alihkan ke wadah karantina afkir / sanitasi khusus",
            "sni_action": "Dilarang diedarkan langsung (SNI 3926:2023 Tabel 1). Noda kotoran melebihi toleransi maksimal 1/16 luas permukaan.",
            "reason": f"Noda kotoran kerabang {dirt_percentage:.1f}% melebihi ambang batas toleransi SNI 3926:2023 Tabel 1 (maksimal 6.25%)",
            "standard_reference": "SNI 3926:2023 Tabel 1 (Toleransi Kebersihan Kerabang)",
            "size_info": f"{size_cat} ({weight_g:.1f}g)"
        }

    # ==========================================================================
    # ATURAN 2: EVALUASI MUTU II (GRADE B - KOMERSIAL / OLAHAN PANGAN)
    # ==========================================================================
    # Cangkang bernoda kotoran ringan (dirt <= 1/32) ATAU Bentuk abnormal (SI < 72 atau > 76)
    is_dirty = (yolo_defect_class == "dirt")
    is_abnormal_shape = (si < 72.0 or si > 76.0)

    if is_dirty or is_abnormal_shape:
        reasons = []
        if is_dirty:
            reasons.append("Kerabang terdapat noda kotoran ringan (toleransi <= 1/32 luas permukaan SNI Tabel 1)")
        if is_abnormal_shape:
            reasons.append(f"Indeks bentuk {si:.1f}% di luar standar ideal SNI 72-76% (agak lonjong/bulat)")

        action = "Gate 2: Alihkan ke jalur komersial / industri olahan pangan (bakery/mie)"

        return {
            "final_grade": "Mutu II (Grade B)",
            "grade_code": "MUTU_II",
            "yolo_primary_factor": f"Kerabang: {yolo_defect_class} ({yolo_confidence:.1%})",
            "sorting_action": action,
            "sni_action": "Lolos konsumsi reguler & bahan baku industri olahan pangan (bakery/kue/mie). Sanitasi kering noda kerabang <= 1/32 (Tabel 1 SNI 3926:2023). Prioritaskan distribusi cepat.",
            "reason": "; ".join(reasons),
            "standard_reference": "SNI 3926:2023 Tabel 1 Syarat Mutu II",
            "size_info": f"{size_cat} ({weight_g:.1f}g)"
        }

    # ==========================================================================
    # ATURAN 3: EVALUASI MUTU I (GRADE A - SEGAR SUPERIOR)
    # ==========================================================================
    # Kerabang utuh bersih mulus & bentuk normal proporsional
    return {
        "final_grade": "Mutu I (Grade A)",
        "grade_code": "MUTU_I",
        "yolo_primary_factor": f"Kerabang utuh & bersih mulus (YOLOv8: good {yolo_confidence:.1%})",
        "sorting_action": f"Gate 1: Kemas ke wadah ritel resmi berlabel ({size_cat})",
        "sni_action": "Layak konsumsi langsung segar (Pasal 8 & 9 SNI 3926:2023). Kemas dalam kemasan ritel berlabel resmi (nama produk, tanggal, NKV, tingkatan mutu). Simpan pada suhu ruang maks 30°C / kelembaban maks 90% (Pasal 10).",
        "reason": f"Kerabang utuh bersih mulus, Shape Index ideal ({si:.1f}%), bobot {weight_g:.1f}g ({size_cat})",
        "standard_reference": "SNI 3926:2023 Tabel 1 Syarat Mutu I",
        "size_info": f"{size_cat} ({weight_g:.1f}g)"
    }
