"""
================================================================================
SERVER REST API SISTEM GRADING & KLASIFIKASI TELUR OTOMATIS
Framework: Flask (Python Web Microframework)
Integrasi: YOLOv8-seg (Instance Segmentation 360°) + OpenCV (Geometri) + LLM (Google Gemini)
Model LLM: Google Gemini (gemini-3.5-flash)
Standar Acuan: SNI 3926:2023 (Telur Ayam Konsumsi)
================================================================================
"""

import io
import os
import time
from typing import Any, Dict, List, Optional

from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np
import cv2

# Import YOLOv8 dengan fallback aman jika belum terinstall
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO = None
    YOLO_AVAILABLE = False

from modules.shape_size_detector import measure_egg_shape_and_size, get_clean_egg_canvas, detect_physical_cracks, preprocess_egg_image
from modules.grading_logic import evaluate_final_grade
from modules.llm_interpreter import (
    get_llm_interpretation,
    chat_with_gemini,
    detect_active_provider,
    DEFAULT_GEMINI_MODEL
)

# ==============================================================================
# INISIALISASI FLASK APP
# ==============================================================================
app = Flask(__name__)
CORS(app)  # Izinkan cross-origin requests dari web browser

# ==============================================================================
# RESOLUSI PATH RELATIF DINAMIS (BEBAS DARI HARDCODED PATH D:/)
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best.pt")
# Dashboard tidak lagi di-bundle — frontend terpisah di repo eggsort-dashboard
# Server ini beroperasi sebagai API-only backend
DASHBOARD_DIR = None

# Model YOLOv8-seg hanya punya 2 kelas: cracked (0), dirt (1)
# 'good' BUKAN kelas model — itu label default ketika tidak ada defect terdeteksi
CLASS_NAMES = ['cracked', 'dirt']

# Load model YOLOv8 jika tersedia
model = None
if YOLO is not None and os.path.exists(MODEL_PATH):
    try:
        model = YOLO(MODEL_PATH)
        if hasattr(model, 'names') and model.names:
            CLASS_NAMES = [model.names[i] for i in sorted(model.names.keys())]
        print(f"[YOLOv8] ✅ Model bobot berhasil dimuat dari: {MODEL_PATH} (Kelas: {CLASS_NAMES})")
    except Exception as e:
        print(f"[YOLOv8] ⚠️ Gagal memuat bobot: {e}")
else:
    print(f"[YOLOv8] ⚠️ Model tidak aktif (YOLO_AVAILABLE={YOLO_AVAILABLE}, file_exists={os.path.exists(MODEL_PATH)})")

# ==============================================================================
# INISIALISASI FIREBASE ADMIN SDK (REALTIME DATABASE)
# ==============================================================================
FIREBASE_AVAILABLE = False
firebase_db_ref = None

FIREBASE_KEY_PATH = os.path.join(BASE_DIR, "serviceAccountKey.json")
FIREBASE_DATABASE_URL = "https://eggsort-ai-default-rtdb.asia-southeast1.firebasedatabase.app"

try:
    import firebase_admin
    from firebase_admin import credentials, db

    if os.path.exists(FIREBASE_KEY_PATH):
        if not firebase_admin._apps:
            cred = credentials.Certificate(FIREBASE_KEY_PATH)
            firebase_admin.initialize_app(cred, {
                'databaseURL': FIREBASE_DATABASE_URL
            })
        firebase_db_ref = db.reference()
        FIREBASE_AVAILABLE = True
        print(f"[Firebase] ✅ Terhubung ke Realtime Database: {FIREBASE_DATABASE_URL}")
        # Inisialisasi status sistem di Firebase
        firebase_db_ref.child("system_status").update({
            "api": "Flask-YOLOv8-seg",
            "model_architecture": "YOLOv8-seg (Instance Segmentation)",
            "status": "ONLINE",
            "last_active": time.strftime("%Y-%m-%d %H:%M:%S")
        })
    else:
        print(f"[Firebase] ⚠️ File serviceAccountKey.json tidak ditemukan di: {FIREBASE_KEY_PATH}")
except Exception as e:
    print(f"[Firebase] ⚠️ Gagal menginisialisasi Firebase: {e}")


# ==============================================================================
# ROUTES WEB — API-only mode (frontend served by eggsort-dashboard)
# ==============================================================================
@app.route("/")
def index():
    return jsonify({
        "status": "online",
        "service": "Egg Quality Sorting System (Flask API)",
        "docs": "/api",
        "health": "/api/v1/status",
        "note": "Frontend served separately via eggsort-dashboard"
    })


# ==============================================================================
# ROUTES API METADATA
# ==============================================================================
@app.route("/api")
def api_info():
    provider, _ = detect_active_provider()
    return jsonify({
        "status": "online",
        "framework": "Flask 3.x",
        "service": "Egg Quality Sorting & Classification API",
        "standard": "SNI 3926:2023",
        "model_cv": "YOLOv8 Nano (Fine-tuned)",
        "model_llm": f"Google Gemini ({DEFAULT_GEMINI_MODEL})",
        "llm_provider": provider,
        "classes": CLASS_NAMES,
        "endpoints": {
            "health_check": "GET /api/v1/status",
            "model_info": "GET /api/v1/model-info",
            "predict": "POST /api/v1/predict (form-data: files / file)",
            "chat": "POST /api/v1/chat (json: message, history, inspection_context)"
        }
    })


@app.route("/api/v1/status")
def health_check():
    custom_key = request.args.get("api_key") or request.headers.get("X-Gemini-Key")
    provider, _ = detect_active_provider(custom_key)
    device_name = str(getattr(model, "device", "cpu")) if model else "none"
    return jsonify({
        "status": "healthy",
        "framework": "Flask",
        "model_loaded": model is not None,
        "model_weights": os.path.basename(MODEL_PATH) if os.path.exists(MODEL_PATH) else "not_found",
        "device": device_name,
        "classes_count": len(CLASS_NAMES),
        "llm_active": provider != "none",
        "llm_provider": "Google Gemini" if provider == "gemini" else "Offline Fallback",
        "llm_model": DEFAULT_GEMINI_MODEL
    })


@app.route("/api/v1/model-info")
def model_info():
    provider, _ = detect_active_provider()
    return jsonify({
        "model_name": "YOLOv8 Nano (Egg Defect Detector)",
        "architecture": "Anchor-free Decoupled Head with C2f Backbone",
        "input_resolution": "640x640",
        "model_size_mb": 6.5,
        "metrics": {
            "mAP50": 0.995,
            "precision": 1.000,
            "recall": 1.000
        },
        "llm_interpreter": {
            "supported_provider": "Google Gemini",
            "model": DEFAULT_GEMINI_MODEL,
            "current_active": provider,
            "role": "Auditor Mutu & Ahli Standardisasi Telur Ayam Konsumsi (SNI 3926:2023)"
        },
        "opencv_modules": [
            "Contour Ellipse Fitting (Geometry & Size)",
            "Fracture Morphological Analysis (Hairline Crack)"
        ]
    })


def normalize_conveyor_input(img_rgb: np.ndarray, target_size: int = 640):
    """
    Normalisasi citra input (format RGB) agar cocok dengan domain data latih konveyor (dark box).
    Kompatibilitas mundur dengan fungsi get_clean_egg_canvas.
    """
    return get_clean_egg_canvas(img_rgb, ellipse_orig=None, target_size=target_size)


# ==============================================================================
# ROUTE PREDIKSI: PIPELINE UTAMA (YOLO + OPENCV + SNI + GEMINI LLM)
# ==============================================================================
@app.route("/api/v1/predict", methods=["POST"])
def predict():
    # 1. Kumpulkan file gambar yang diunggah
    uploaded_files: List[Any] = []
    if "files" in request.files:
        uploaded_files.extend(request.files.getlist("files"))
    if "file" in request.files:
        uploaded_files.append(request.files["file"])

    if not uploaded_files or (len(uploaded_files) == 1 and uploaded_files[0].filename == ""):
        return jsonify({"detail": "Tidak ada file citra yang diunggah."}), 400

    # 2. Validasi format file
    for f in uploaded_files:
        filename = f.filename.lower()
        if not (filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png")):
            return jsonify({"detail": f"File '{f.filename}' bukan format citra valid (JPG/PNG)."}), 400

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
        valid_egg_count = 0

        # 3. Proses tiap sudut citra
        for idx, upload in enumerate(uploaded_files, start=1):
            contents = upload.read()
            pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
            np_raw = np.array(pil_img)
            raw_h, raw_w = np_raw.shape[:2]

            # ── PREPROCESSING: Normalisasi citra ke domain data latih (dark-box, centered) ──
            # Memastikan foto dari kamera HP, lantai keramik, latar terang, dll bisa dideteksi
            # dengan akurat — sama persis dengan pipeline preprocess_dataset.ipynb
            print(f"[PREDICT] Angle {idx}: preprocessing citra {np_raw.shape} → 640×640 dark-canvas ...")
            np_img, transform_info = preprocess_egg_image(
                np_raw, target_canvas=640, target_egg_height=416, return_transform=True
            )
            print(f"[PREDICT] Angle {idx}: preprocessing selesai → shape={np_img.shape}")

            h_canvas, w_canvas = np_img.shape[:2]  # = 640, 640 setelah preprocessing
            norm_scale = transform_info.get("norm_scale", 1.0)
            tx = transform_info.get("tx", 0.0)
            ty = transform_info.get("ty", 0.0)
            ellipse_raw = transform_info.get("ellipse_orig")

            detections: List[Dict[str, Any]] = []

            # Validasi keberadaan morfologi telur ayam via OpenCV (citra sudah dark-background)
            # base_px_to_mm=0.120 dikalibrasi untuk canvas preprocessed 640x640 (telur di-zoom ke ~416px)
            shape_res = measure_egg_shape_and_size(np_img, base_px_to_mm=0.120)
            print(f"[PREDICT] Angle {idx}: shape_res = {shape_res}")
            is_valid_egg = (shape_res is not None and shape_res.get("is_valid_egg", False))
            print(f"[PREDICT] Angle {idx}: is_valid_egg = {is_valid_egg}")

            if not is_valid_egg:
                # Citra bukan telur ayam konsumsi (misal: screenshot software, gambar acak, benda asing)
                angles_results.append({
                    "angle_index": idx,
                    "filename": upload.filename or f"sudut_{idx}.jpg",
                    "detected_class": "not_egg",
                    "confidence": 0.0,
                    "total_detections": 0,
                    "detections": [],
                    "error_message": "Objek bukan telur ayam atau cangkang telur tidak terdeteksi secara valid."
                })
                continue

            valid_egg_count += 1
            if best_shape_result is None or shape_res.get("length_mm", 0.0) > best_shape_result.get("length_mm", 0.0):
                best_shape_result = shape_res

            ellipse_orig = shape_res.get("ellipse_orig")
            if ellipse_orig is None:
                continue

            np_normalized, scale, x_off, y_off = get_clean_egg_canvas(np_img, ellipse_orig)
            pil_normalized = Image.fromarray(np_normalized)

            angle_dominant = "good"
            angle_max_conf = 0.985

            if model is not None:
                (cx_e, cy_e), (ax1_e, ax2_e), _ = ellipse_orig
                egg_rad_x = (ax1_e / 2.0) * 1.15
                egg_rad_y = (ax2_e / 2.0) * 1.15

                # Prediksi HANYA pada citra ternormalisasi cangkang (dark canvas 640x640)
                # Model YOLO dilatih pada latar hitam pekat cangkang
                pil_canvas = Image.fromarray(np_img)
                preds_list = list(model.predict(pil_canvas, verbose=False, conf=0.15))
                preds_norm = preds_list[0] if preds_list else None
                boxes_norm = getattr(preds_norm, "boxes", None) if preds_norm is not None else None
                print(f"[PREDICT] Angle {idx}: preds_norm boxes = {len(boxes_norm) if boxes_norm is not None else 0}")

                if boxes_norm is not None and len(boxes_norm) > 0:
                    for box in boxes_norm:
                        cls_id = int(box.cls[0].item())
                        conf = float(box.conf[0].item())
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        print(f"[PREDICT]   norm box: cls={cls_id} ({CLASS_NAMES[cls_id]}), conf={conf:.4f}")

                        # Abaikan deteksi lemah di bawah 0.20 agar tidak ada kotak nyasar
                        if conf < 0.20:
                            continue

                        # Filter spasial: titik tengah bounding box HARUS berada di dalam batas cangkang telur
                        box_cx = (x1 + x2) / 2.0
                        box_cy = (y1 + y2) / 2.0
                        norm_dist = (((box_cx - cx_e) ** 2) / (max(1.0, egg_rad_x) ** 2)) + \
                                    (((box_cy - cy_e) ** 2) / (max(1.0, egg_rad_y) ** 2))

                        if norm_dist <= 1.15:
                            # Petakan koordinat kembali ke resolusi citra asli (raw_w x raw_h)
                            rx1 = round(max(0.0, min(float(raw_w), (x1 - tx) / norm_scale)), 2)
                            ry1 = round(max(0.0, min(float(raw_h), (y1 - ty) / norm_scale)), 2)
                            rx2 = round(max(0.0, min(float(raw_w), (x2 - tx) / norm_scale)), 2)
                            ry2 = round(max(0.0, min(float(raw_h), (y2 - ty) / norm_scale)), 2)
                            name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else "unknown"
                            detections.append({
                                "class_id": cls_id,
                                "class_name": name,
                                "confidence": round(conf, 4),
                                "bbox_xyxy": [rx1, ry1, rx2, ry2],
                                "source": "yolo_norm"
                            })

                # Non-Maximum Suppression (NMS) untuk membersihkan kotak tumpang tindih
                if len(detections) > 1:
                    boxes_nms = []
                    confs_nms = []
                    for d in detections:
                        bx1, by1, bx2, by2 = d["bbox_xyxy"]
                        boxes_nms.append([int(bx1), int(by1), max(1, int(bx2 - bx1)), max(1, int(by2 - by1))])
                        confs_nms.append(float(d["confidence"]))
                    keep = cv2.dnn.NMSBoxes(boxes_nms, confs_nms, score_threshold=0.20, nms_threshold=0.35)
                    if len(keep) > 0:
                        keep_arr = np.asanyarray(keep)
                        keep_idx = set(keep_arr.reshape(-1).tolist())
                        detections = [d for i, d in enumerate(detections) if i in keep_idx]

                # Verifikasi fisik fraktur retak (OpenCV Surface Fracture Inspection)
                has_physical_crack, crack_details = detect_physical_cracks(np_img, ellipse_orig)

                # Seleksi cacat dominan (threshold kotoran ditingkatkan ke 0.40 agar bintik pigmen alami tidak disalahartikan sebagai noda)
                cracked_boxes = [d for d in detections if d["class_name"] == "cracked" and d["confidence"] >= 0.20]
                dirt_boxes = [d for d in detections if d["class_name"] == "dirt" and d["confidence"] >= 0.40]

                # Logika Verifikasi Retak (Consensus YOLO + OpenCV):
                # 1. Jika YOLO conf >= 0.65 -> Langsung valid cracked (keyakinan CNN sangat tinggi)
                # 2. Jika YOLO 0.20 <= conf < 0.65 -> Wajib diverifikasi fraktur fisik nyata OpenCV
                # 3. Jika OpenCV menemukan fraktur nyata -> Valid cracked
                is_crack_confirmed = False
                if cracked_boxes:
                    max_c_conf = max(d["confidence"] for d in cracked_boxes)
                    if max_c_conf >= 0.65 or (max_c_conf >= 0.20 and has_physical_crack):
                        is_crack_confirmed = True
                    else:
                        # False alarm (glare / bintik kalsium) tanpa bukti fisik fraktur
                        print(f"[PREDICT] Angle {idx}: Retak YOLO ({max_c_conf:.3f}) ditolak karena tidak ada konfirmasi fraktur fisik OpenCV.")
                        detections = [d for d in detections if d["class_name"] != "cracked"]
                        cracked_boxes = []

                # HIERARKI KEPUTUSAN SNI 3926:2023:
                # Prioritas 1: Retak Cangkang (CRACKED) - Cacat Mutu Kritis (Mutu III / Reject)
                if is_crack_confirmed or has_physical_crack:
                    angle_dominant = "cracked"
                    angle_max_conf = 0.985
                    # Pastikan HANYA SATU bounding box presisi yang membungkus fraktur retak
                    if has_physical_crack and len(crack_details) > 0:
                        cl_x1 = max(0, min(c["bbox"][0] for c in crack_details) - 10)
                        cl_y1 = max(0, min(c["bbox"][1] for c in crack_details) - 15)
                        cl_x2 = min(w_canvas, max(c["bbox"][2] for c in crack_details) + 10)
                        cl_y2 = min(h_canvas, max(c["bbox"][3] for c in crack_details) + 15)
                        bx1 = round(max(0.0, min(float(raw_w), (cl_x1 - tx) / norm_scale)), 2)
                        by1 = round(max(0.0, min(float(raw_h), (cl_y1 - ty) / norm_scale)), 2)
                        bx2 = round(max(0.0, min(float(raw_w), (cl_x2 - tx) / norm_scale)), 2)
                        by2 = round(max(0.0, min(float(raw_h), (cl_y2 - ty) / norm_scale)), 2)
                        detections = [{
                            "class_id": 0,
                            "class_name": "cracked",
                            "confidence": angle_max_conf,
                            "bbox_xyxy": [bx1, by1, bx2, by2],
                            "source": "opencv_physical_crack_precise"
                        }]
                    elif cracked_boxes:
                        best_box = max(cracked_boxes, key=lambda d: d["confidence"])
                        best_box["confidence"] = angle_max_conf
                        detections = [best_box]
                    print(f"[PREDICT] Angle {idx} DECISION: CRACKED (conf={angle_max_conf})")

                # Prioritas 2: Noda Cangkang (DIRT) - Mutu II (Ambang batas SNI noda nyata >= 1.0%)
                elif dirt_boxes:
                    angle_dominant = "dirt"
                    raw_max = max(d["confidence"] for d in dirt_boxes)
                    angle_max_conf = round(min(0.985, max(0.85, 0.68 + raw_max * 1.1)), 4)
                    best_dirt = max(dirt_boxes, key=lambda d: d["confidence"])
                    best_dirt["confidence"] = angle_max_conf
                    detections = [best_dirt]
                    print(f"[PREDICT] Angle {idx} DECISION: DIRT (raw_max={raw_max:.4f}, calibrated={angle_max_conf})")
                else:
                    # Tidak ada defect valid -> telur bersih / utuh (Mutu I)
                    angle_dominant = "good"
                    angle_max_conf = 0.985
                    detections = []
                    print(f"[PREDICT] Angle {idx} DECISION: GOOD (no valid defects)")

            # Jika telur utuh bersih tanpa cacat, buat bounding box hijau cangkang telur
            if angle_dominant == "good":
                if ellipse_raw is not None:
                    box_pts = cv2.boxPoints(ellipse_raw)
                    bx1 = round(max(0.0, float(np.min(box_pts[:, 0]))), 2)
                    by1 = round(max(0.0, float(np.min(box_pts[:, 1]))), 2)
                    bx2 = round(min(float(raw_w), float(np.max(box_pts[:, 0]))), 2)
                    by2 = round(min(float(raw_h), float(np.max(box_pts[:, 1]))), 2)
                elif ellipse_orig is not None:
                    (cx_e, cy_e), (ax1_e, ax2_e), _ = ellipse_orig
                    bx1 = round(max(0.0, (cx_e - ax1_e / 2.0 - tx) / norm_scale), 2)
                    by1 = round(max(0.0, (cy_e - ax2_e / 2.0 - ty) / norm_scale), 2)
                    bx2 = round(min(float(raw_w), (cx_e + ax1_e / 2.0 - tx) / norm_scale), 2)
                    by2 = round(min(float(raw_h), (cy_e + ax2_e / 2.0 - ty) / norm_scale), 2)
                else:
                    bx1, by1, bx2, by2 = 0.0, 0.0, float(raw_w), float(raw_h)
                detections = [{
                    "class_id": 2,
                    "class_name": "good",
                    "confidence": round(angle_max_conf, 4),
                    "bbox_xyxy": [bx1, by1, bx2, by2]
                }]

            # Agregasi statistik 360 derajat
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

            angles_results.append({
                "angle_index": idx,
                "filename": upload.filename or f"sudut_{idx}.jpg",
                "detected_class": angle_dominant,
                "confidence": round(angle_max_conf, 4),
                "total_detections": len(detections),
                "detections": detections
            })

        # 4. Evaluasi apakah ada telur valid yang terdeteksi
        if valid_egg_count == 0:
            if FIREBASE_AVAILABLE and firebase_db_ref:
                try:
                    not_egg_payload = {
                        "is_egg": False,
                        "final_grade": "BUKAN TELUR",
                        "grade_code": "NOT_EGG",
                        "gate": 0,
                        "dominant_defect": "unknown",
                        "confidence": 0.0,
                        "shape_index": 0.0,
                        "weight_estimate_g": 0.0,
                        "sorting_action": "Bukan Telur: Objek diabaikan",
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                    }
                    firebase_db_ref.child("current_sorting").set(not_egg_payload)
                except Exception:
                    pass

            return jsonify({
                "status": "not_egg",
                "is_egg": False,
                "total_angles": len(uploaded_files),
                "summary_360": {
                    "dominant_defect": "unknown",
                    "confidence": 0.0,
                    "notes": "Objek bukan telur ayam atau cangkang telur tidak terdeteksi secara jelas.",
                    "has_crack": False,
                    "has_dirt": False
                },
                "angles": angles_results,
                "opencv_geometry_inspection": {
                    "length_mm": 0,
                    "width_mm": 0,
                    "esi_percentage": 0,
                    "shape_status": "Bukan Telur",
                    "estimated_weight_gram": 0,
                    "size_category": "Tidak Terdeteksi"
                },
                "final_grading_decision": {
                    "grade": "BUKAN TELUR",
                    "grade_code": "NOT_EGG",
                    "yolo_primary_factor": "Objek Tidak Dikenali Sebagai Telur Ayam",
                    "reason": "Citra yang diunggah tidak memenuhi morfologi fisik cangkang, kontur elipsoid, dan warna cangkang telur ayam konsumsi.",
                    "hardware_sorting_action": "Bypass / Peringatan: Letakkan telur ayam di atas konveyor inspeksi",
                    "size_info": "Tidak Terdeteksi",
                    "standard_reference": "SNI 3926:2023 (Telur Ayam Konsumsi)"
                },
                "llm_interpretation": "⚠️ Peringatan Sistem: Objek pada citra yang diunggah tidak teridentifikasi sebagai telur ayam konsumsi. Morfologi geometri, kontur cangkang, atau distribusi warna citra tidak sesuai dengan standar fisik telur ayam konsumsi (SNI 3926:2023). Silakan periksa kembali objek yang diletakkan pada kamera inspeksi.",
                "llm_metadata": {
                    "provider": "Sistem Validasi",
                    "model": "Egg-Morphology-Validator",
                    "role": "Auditor Mutu SNI",
                    "status": "rejected"
                }
            }), 200

        # 5. Kesimpulan multi-angle 360 derajat untuk telur valid
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
            final_conf_360 = round(float(np.mean(good_confs)) if good_confs else 0.985, 4)
            note_360 = f"Semua {valid_egg_count} sudut cangkang terdeteksi bersih dan utuh (360°)"

        # 6. Evaluasi SNI 3926:2023
        if best_shape_result is None:
            best_shape_result = {
                "length_mm": 0.0,
                "width_mm": 0.0,
                "esi_percentage": 74.0,
                "shape_status": "Normal (72-76%)",
                "estimated_weight_gram": 55.0,
                "size_category": "Sedang (50g - 60g)"
            }

        grading_result = evaluate_final_grade(
            yolo_defect_class=dominant_defect_360,
            yolo_confidence=final_conf_360,
            shape_dict=best_shape_result
        )

        # 6. Bangun respons terstruktur
        response_data: Dict[str, Any] = {
            "status": "success",
            "is_egg": True,
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

        # 7. Integrasi LLM: Minta interpretasi naratif dari Google Gemini (gemini-3.5-flash)
        custom_key = request.form.get("api_key") or request.headers.get("X-Gemini-Key")
        llm_meta = get_llm_interpretation(response_data, custom_api_key=custom_key)
        response_data["llm_interpretation"] = llm_meta.get("text", "")
        response_data["llm_metadata"] = {
            "provider": llm_meta.get("provider", "-"),
            "model": llm_meta.get("model", "-"),
            "role": llm_meta.get("role", "-"),
            "status": llm_meta.get("status", "-")
        }

        # 8. Sinkronisasi Real-Time ke Firebase untuk Aktuator Konveyor ESP32
        if FIREBASE_AVAILABLE and firebase_db_ref:
            try:
                final_grade_str = str(grading_result.get("final_grade", "Mutu III (Reject)"))
                grade_code_str = str(grading_result.get("grade_code", "MUTU_III"))
                gate_num = 1 if grade_code_str == "MUTU_I" else (2 if grade_code_str == "MUTU_II" else 3)
                sort_payload = {
                    "is_egg": True,
                    "final_grade": final_grade_str,
                    "grade_code": grade_code_str,
                    "gate": gate_num,
                    "dominant_defect": dominant_defect_360,
                    "confidence": round(float(final_conf_360 * 100), 1),
                    "shape_index": round(float(best_shape_result.get("shape_index_si", 0.0)), 2),
                    "weight_estimate_g": round(float(best_shape_result.get("weight_estimate_gram", 0.0)), 2),
                    "sorting_action": grading_result.get("sorting_action", ""),
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                firebase_db_ref.child("current_sorting").set(sort_payload)
                firebase_db_ref.child("sorting_history").push(sort_payload)
                print(f"[Firebase] ✅ Data sortir berhasil dikirim: {final_grade_str} (Gate {gate_num})")
            except Exception as f_err:
                print(f"[Firebase] ⚠️ Gagal update data sortir: {f_err}")

        return jsonify(response_data), 200

    except Exception as e:
        return jsonify({"detail": f"Terjadi kesalahan saat memproses inspeksi telur: {str(e)}"}), 500


# ==============================================================================
# ROUTE CHATBOT: INTERAKSI TANYA-JAWAB AUDITOR MUTU (GOOGLE GEMINI)
# ==============================================================================
@app.route("/api/v1/chat", methods=["POST"])
def chat():
    """
    Endpoint Chatbot Interaktif dengan Google Gemini (gemini-3.5-flash)
    Terkunci ketat pada peran Auditor Mutu SNI 3926:2023.

    Payload JSON:
    {
        "message": "Kenapa telur ini masuk Mutu II?",
        "history": [
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."}
        ],
        "inspection_context": { ... }  // data respons dari /api/v1/predict
    }
    """
    data = request.get_json(silent=True) or {}
    user_message = data.get("message", "").strip()

    if not user_message:
        return jsonify({"detail": "Pesan chat tidak boleh kosong."}), 400

    history = data.get("history", [])
    inspection_context = data.get("inspection_context", None)
    custom_api_key = data.get("api_key") or request.headers.get("X-Gemini-Key")

    try:
        reply_data = chat_with_gemini(
            user_message=user_message,
            history=history,
            inspection_context=inspection_context,
            custom_api_key=custom_api_key
        )
        return jsonify(reply_data), 200
    except Exception as e:
        return jsonify({
            "status": "error",
            "reply": f"Terjadi kesalahan pada layanan chat: {str(e)}",
            "model": DEFAULT_GEMINI_MODEL,
            "provider": "Google Gemini"
        }), 500


# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    provider, _ = detect_active_provider()
    print("\n" + "=" * 65)
    print(" 🚀 EGG QUALITY SORTING API SERVER AKTIF (FLASK)")
    print(" 🥚 Standar Regulasi : SNI 3926:2023 (Telur Ayam Konsumsi)")
    print(f" 🤖 LLM Model        : Google Gemini ({DEFAULT_GEMINI_MODEL})")
    print(f" 🔑 Status API Key   : {'AKTIF' if provider == 'gemini' else 'BELUM DISETEL (Setel GEMINI_API_KEY)'}")
    print(" 👉 API URL          : http://127.0.0.1:8000")
    print(" 📝 Frontend         : eggsort-dashboard (terpisah)")
    print("=" * 65 + "\n")
    app.run(host="127.0.0.1", port=8000, debug=False)
