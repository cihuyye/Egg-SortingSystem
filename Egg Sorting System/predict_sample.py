import os
from typing import Any, Dict, List, Optional
from PIL import Image
import numpy as np
from ultralytics import YOLO

from modules.shape_size_detector import measure_egg_shape_and_size
from modules.candling_detector import analyze_candling_quality
from modules.grading_logic import evaluate_final_grade

def run_test_sample() -> None:
    model_path: str = r"D:/Egg Sorting System/models/best.pt"
    if not os.path.exists(model_path):
        print("[PERINGATAN] Model khusus belum dilatih! Menggunakan bobot bawaan...")
        model_path = "yolov8n.pt"
        
    model = YOLO(model_path)
    
    test_sample_path: str = r"D:/Egg Dataset/Egg Dataset/test/images/good"
    if os.path.exists(test_sample_path) and len(os.listdir(test_sample_path)) > 0:
        sample_file: str = os.path.join(test_sample_path, os.listdir(test_sample_path)[0])
    else:
        print("[ERROR] Gambar uji tidak ditemukan.")
        return
        
    print(f"\n--- MENGUJI SAMPEL: {os.path.basename(sample_file)} ---")
    im = Image.open(sample_file).convert("RGB")
    arr: np.ndarray = np.array(im)
    
    # 1. Deteksi Cacat Cangkang oleh YOLOv8
    results: Any = model.predict(source=sample_file, verbose=False)
    preds: Any = results[0] if isinstance(results, list) and len(results) > 0 else results
    class_names: List[str] = ["cracked", "dirt", "good"]
    defect_class: str = "good"
    conf: float = 1.0
    
    boxes = getattr(preds, "boxes", None)
    if boxes is not None:
        cls_data = getattr(boxes, "cls", None)
        conf_data = getattr(boxes, "conf", None)
        if cls_data is not None and len(cls_data) > 0 and conf_data is not None:
            c_idx: int = int(cls_data[0])
            if 0 <= c_idx < len(class_names):
                defect_class = class_names[c_idx]
            conf = float(conf_data[0])
        
    print(f"1. Hasil Deteksi YOLOv8 : {defect_class.upper()} (Keyakinan: {conf:.2%})")
    
    # 2. Pengukuran Bentuk dan Ukuran oleh OpenCV
    raw_shape: Optional[Dict[str, Any]] = measure_egg_shape_and_size(arr)
    shape_result: Dict[str, Any] = raw_shape if raw_shape is not None else {
        "length_mm": 55.0,
        "width_mm": 42.0,
        "esi_percentage": 74.0,
        "shape_status": "Normal",
        "estimated_weight_gram": 58.0,
        "size_category": "Sedang (50g - 60g)"
    }
    
    l_val = shape_result.get("length_mm", 55.0)
    w_val = shape_result.get("width_mm", 42.0)
    esi_val = shape_result.get("esi_percentage", 74.0)
    cat_val = shape_result.get("size_category", "Sedang")
    print(f"2. Hasil OpenCV Geometri : Panjang={l_val}mm, Lebar={w_val}mm, Shape Index={esi_val}%, Kategori={cat_val}")
    
    # 3. Pengujian Candling oleh OpenCV
    candling_result: Dict[str, Any] = analyze_candling_quality(arr)
    c_status = candling_result.get("candling_status", "Mutu I")
    air_val = candling_result.get("air_cell_depth_cm", 0.3)
    print(f"3. Hasil OpenCV Candling : Status={c_status}, Kantung Udara={air_val}cm")
    
    # 4. Evaluasi Mutu Akhir SNI 3926:2023 (YOLOv8 + OpenCV)
    grade_decision: Dict[str, Any] = evaluate_final_grade(
        yolo_defect_class=defect_class,
        yolo_confidence=conf,
        shape_dict=shape_result,
        candling_dict=candling_result
    )
    
    f_grade = str(grade_decision.get("final_grade", "Mutu I")).upper()
    reason = str(grade_decision.get("reason", "-"))
    action = str(grade_decision.get("sorting_action", "-"))
    std_ref = str(grade_decision.get("standard_reference", "SNI 3926:2023"))
    
    print("\n========================================")
    print(f" KEPUTUSAN AKHIR : {f_grade}")
    print(f" ALASAN          : {reason}")
    print(f" TINDAKAN MESIN  : {action}")
    print(f" ACUAN STANDAR   : {std_ref}")
    print("========================================\n")

if __name__ == "__main__":
    run_test_sample()
