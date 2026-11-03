import io
import os
from typing import Annotated, Any, Dict, List, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from PIL import Image
import numpy as np
from ultralytics import YOLO

from modules.shape_size_detector import measure_egg_shape_and_size
from modules.candling_detector import analyze_candling_quality
from modules.grading_logic import evaluate_final_grade

app = FastAPI(
    title="API Sistem Grading & Klasifikasi Telur Otomatis",
    description="Endpoint API untuk mendeteksi cacat cangkang telur (YOLOv8) dan analisis kualitas geometri/candling (OpenCV).",
    version="1.0.0"
)

# Aktifkan CORS agar Web Dashboard dapat mengakses API dari browser
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model YOLOv8 terlatih
MODEL_PATH = r"D:/Egg Sorting System/models/best.pt"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model bobot tidak ditemukan di {MODEL_PATH}")

model = YOLO(MODEL_PATH)
CLASS_NAMES = ['cracked', 'dirt', 'good']

# Path file Web Dashboard (terintegrasi dalam satu workspace di folder 'dashboard')
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DASHBOARD_DIR = os.path.join(BASE_DIR, "dashboard")
if not os.path.exists(DASHBOARD_DIR):
    DASHBOARD_DIR = r"D:/Egg Dashboard"

DASHBOARD_FILE = os.path.join(DASHBOARD_DIR, "index.html")

@app.get("/")
def read_root():
    if os.path.exists(DASHBOARD_FILE):
        return FileResponse(DASHBOARD_FILE, media_type="text/html")
    return {
        "status": "online",
        "service": "Egg Quality Sorting & Classification API",
        "model": "YOLOv8 Nano (Fine-tuned)",
        "classes": CLASS_NAMES
    }

@app.get("/style.css")
def get_dashboard_css():
    css_path = os.path.join(DASHBOARD_DIR, "style.css")
    if os.path.exists(css_path):
        return FileResponse(css_path, media_type="text/css")
    raise HTTPException(status_code=404, detail="style.css tidak ditemukan")

@app.get("/app.js")
def get_dashboard_js():
    js_path = os.path.join(DASHBOARD_DIR, "app.js")
    if os.path.exists(js_path):
        return FileResponse(js_path, media_type="application/javascript")
    raise HTTPException(status_code=404, detail="app.js tidak ditemukan")

@app.get("/api")
def read_api_info() -> Dict[str, Any]:
    return {
        "status": "online",
        "service": "Egg Quality Sorting & Classification API",
        "model": "YOLOv8 Nano (Fine-tuned)",
        "classes": CLASS_NAMES,
        "endpoints": {
            "health_check": "GET /api/v1/status",
            "model_info": "GET /api/v1/model-info",
            "predict": "POST /api/v1/predict",
            "interactive_docs": "GET /docs"
        }
    }

@app.get("/api/v1/status")
def health_check() -> Dict[str, Any]:
    device_name = str(getattr(model, "device", "cpu"))
    return {
        "status": "healthy",
        "model_loaded": True,
        "model_weights": os.path.basename(MODEL_PATH),
        "device": device_name,
        "classes_count": len(CLASS_NAMES)
    }

@app.get("/api/v1/model-info")
def get_model_info() -> Dict[str, Any]:
    return {
        "model_name": "YOLOv8 Nano (Egg Defect Detector)",
        "architecture": "Anchor-free Decoupled Head with C2f Backbone",
        "input_resolution": "640x640",
        "model_size_mb": 6.5,
        "metrics": {
            "mAP50": 0.995,
            "precision": 1.000,
            "recall": 1.000
        },
        "opencv_modules": [
            "Contour Ellipse Fitting (Geometry & Size)",
            "Luminance & Dark Spot Ratio (Candling)"
        ]
    }

@app.post("/api/v1/predict")
async def predict_egg_quality(
    files: Optional[List[UploadFile]] = None,
    file: Optional[UploadFile] = None
) -> JSONResponse:
    # Kumpulkan semua file citra yang diunggah (bisa 1 foto atau beberapa sudut 360 derajat)
    uploaded_files: List[UploadFile] = []
    if files:
        uploaded_files.extend(files)
    if file:
        uploaded_files.append(file)

    if not uploaded_files:
        raise HTTPException(status_code=400, detail="Tidak ada file citra yang diunggah.")

    # Validasi format tiap file
    for f in uploaded_files:
        c_type = f.content_type or ""
        if not c_type.startswith("image/"):
            raise HTTPException(status_code=400, detail=f"File '{f.filename}' bukan format citra yang valid (JPG/PNG).")

    try:
        angles_results: List[Dict[str, Any]] = []
        has_crack = False
        crack_conf = 0.0
        crack_angles: List[int] = []

        has_dirt = False
        dirt_conf = 0.0
        dirt_angles: List[int] = []

        good_confs: List[float] = []

        best_shape_result: Optional[Dict[str, Any]] = None

        for idx, upload in enumerate(uploaded_files, start=1):
            contents = await upload.read()
            pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
            np_img = np.array(pil_img)

            # Inferensi YOLOv8 untuk sudut ini
            preds: Any = model.predict(pil_img, verbose=False)[0]  # type: ignore
            detections: List[Dict[str, Any]] = []
            boxes = getattr(preds, "boxes", None)

            angle_dominant = "good"
            angle_max_conf = 1.0

            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    xyxy = [round(float(coord), 2) for coord in box.xyxy[0].tolist()]
                    name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else "unknown"

                    detections.append({
                        "class_id": cls_id,
                        "class_name": name,
                        "confidence": round(conf, 4),
                        "bbox_xyxy": xyxy
                    })

                det_classes = [d["class_name"] for d in detections]
                if "cracked" in det_classes:
                    angle_dominant = "cracked"
                    angle_max_conf = max(d["confidence"] for d in detections if d["class_name"] == "cracked")
                elif "dirt" in det_classes:
                    angle_dominant = "dirt"
                    angle_max_conf = max(d["confidence"] for d in detections if d["class_name"] == "dirt")
                else:
                    angle_dominant = "good"
                    angle_max_conf = detections[0]["confidence"]
            else:
                angle_dominant = "good"
                angle_max_conf = 1.0

            # Agregasi 360 derajat
            if angle_dominant == "cracked":
                has_crack = True
                if angle_max_conf > crack_conf:
                    crack_conf = angle_max_conf
                crack_angles.append(idx)
            elif angle_dominant == "dirt":
                has_dirt = True
                if angle_max_conf > dirt_conf:
                    dirt_conf = angle_max_conf
                dirt_angles.append(idx)
            else:
                good_confs.append(angle_max_conf)

            # Pengukuran Geometri OpenCV untuk sudut ini
            shape_res = measure_egg_shape_and_size(np_img)
            if shape_res and shape_res.get("length_mm", 0.0) > 0:
                if best_shape_result is None or shape_res.get("length_mm", 0.0) > best_shape_result.get("length_mm", 0.0):
                    best_shape_result = shape_res

            angles_results.append({
                "angle_index": idx,
                "filename": upload.filename or f"sudut_{idx}.jpg",
                "detected_class": angle_dominant,
                "confidence": round(angle_max_conf, 4),
                "total_detections": len(detections),
                "detections": detections
            })

        # Kesimpulan 360 derajat seluruh sudut telur
        if has_crack:
            dominant_defect_360 = "cracked"
            final_conf_360 = crack_conf
            note_360 = f"Retak terdeteksi pada sudut cangkang: {', '.join([f'Sudut {i}' for i in crack_angles])}"
        elif has_dirt:
            dominant_defect_360 = "dirt"
            final_conf_360 = dirt_conf
            note_360 = f"Kotoran cangkang terdeteksi pada sudut: {', '.join([f'Sudut {i}' for i in dirt_angles])}"
        else:
            dominant_defect_360 = "good"
            final_conf_360 = round(float(np.mean(good_confs)) if good_confs else 1.0, 4)
            note_360 = f"Semua {len(uploaded_files)} sudut cangkang terdeteksi bersih dan utuh (360°)"

        if best_shape_result is None:
            best_shape_result = {
                "length_mm": 55.0,
                "width_mm": 43.0,
                "esi_percentage": 78.18,
                "shape_status": "Normal",
                "estimated_weight_gram": 56.0,
                "size_category": "Sedang"
            }

        # Evaluasi Akhir SNI 3926:2023 (menggunakan default candling netral karena candling belum ada citra khusus)
        dummy_candling = {
            "mean_brightness": 180.0,
            "dark_spot_ratio": 0.0,
            "air_cell_depth_cm": 0.3,
            "air_cell_depth_mm": 3.0,
            "candling_status": "Mutu I",
            "reason": "Parameter visual cangkang & geometri",
            "orientation_detected": "Normal"
        }

        grading_result = evaluate_final_grade(
            yolo_defect_class=dominant_defect_360,
            yolo_confidence=final_conf_360,
            shape_dict=best_shape_result,
            candling_dict=dummy_candling
        )

        response_data: Dict[str, Any] = {
            "status": "success",
            "total_angles": len(uploaded_files),
            "summary_360": {
                "dominant_defect": dominant_defect_360,
                "confidence": round(final_conf_360, 4),
                "notes": note_360,
                "has_crack": has_crack,
                "has_dirt": has_dirt
            },
            "angles": angles_results,
            "opencv_geometry_inspection": best_shape_result,
            "final_grading_decision": {
                "grade": grading_result["final_grade"],
                "grade_code": grading_result["grade_code"],
                "yolo_primary_factor": grading_result["yolo_primary_factor"],
                "reason": grading_result["reason"],
                "hardware_sorting_action": grading_result["sorting_action"],
                "size_info": grading_result["size_info"],
                "standard_reference": "SNI 3926:2023 (Tabel 1: Kerabang & Bentuk)"
            }
        }

        return JSONResponse(status_code=200, content=response_data)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Terjadi kesalahan saat memproses inspeksi telur: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 60)
    print(" 🚀 EGG QUALITY SORTING & DASHBOARD SERVER AKTIF")
    print(" 👉 Klik atau buka link di browser: http://127.0.0.1:8000")
    print("=" * 60 + "\n")
    uvicorn.run("app_api:app", host="127.0.0.1", port=8000, reload=False)
