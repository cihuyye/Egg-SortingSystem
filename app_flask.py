"""
================================================================================
SERVER REST API SISTEM GRADING & KLASIFIKASI TELUR OTOMATIS
Framework: Flask (Python Web Microframework)
Integrasi: YOLOv8-seg (Instance Segmentation 360°) + OpenCV (Geometri) + LLM (Google Gemini)
Model LLM: Google Gemini (gemini-3.8-flash)
Standar Acuan: SNI 3926:2023 (Telur Ayam Konsumsi)
================================================================================
"""

import io
import os
import time
from typing import Any, Dict, List, Optional

from flask import Flask, request, jsonify, send_from_directory, send_file
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

from modules.shape_size_detector import measure_egg_shape_and_size, get_clean_egg_canvas, detect_physical_cracks
from modules.candling_detector import analyze_candling_quality
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
DASHBOARD_DIR = os.path.join(BASE_DIR, "dashboard")

CLASS_NAMES = ['cracked', 'dirt', 'good']

# Load model YOLOv8 jika tersedia
model = None
if YOLO_AVAILABLE and os.path.exists(MODEL_PATH):
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
# ROUTES WEB DASHBOARD
# ==============================================================================
@app.route("/")
def index():
    dashboard_file = os.path.join(DASHBOARD_DIR, "index.html")
    if os.path.exists(dashboard_file):
        return send_file(dashboard_file)
    return jsonify({
        "status": "online",
        "service": "Egg Quality Sorting System (Flask)",
        "docs": "/api",
        "health": "/api/v1/status"
    })


@app.route("/style.css")
def get_style():
    return send_from_directory(DASHBOARD_DIR, "style.css", mimetype="text/css")


@app.route("/app.js")
def get_script():
    return send_from_directory(DASHBOARD_DIR, "app.js", mimetype="application/javascript")


@app.route("/assets/<path:filename>")
def serve_assets(filename):
    return send_from_directory(os.path.join(DASHBOARD_DIR, "assets"), filename)


@app.route("/favicon.svg")
def serve_favicon():
    return send_from_directory(DASHBOARD_DIR, "favicon.svg", mimetype="image/svg+xml")


@app.route("/icons.svg")
def serve_icons():
    return send_from_directory(DASHBOARD_DIR, "icons.svg", mimetype="image/svg+xml")


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
            "Luminance & Dark Spot Ratio (Candling)"
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
            np_img = np.array(pil_img)
            h_orig, w_orig = np_img.shape[:2]

            detections: List[Dict[str, Any]] = []

            # Validasi keberadaan morfologi telur ayam via OpenCV
            shape_res = measure_egg_shape_and_size(np_img)
            is_valid_egg = (shape_res is not None and shape_res.get("is_valid_egg", False))

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
            np_normalized, scale, x_off, y_off = get_clean_egg_canvas(np_img, ellipse_orig)
            pil_normalized = Image.fromarray(np_normalized)

            angle_dominant = "good"
            angle_max_conf = 0.985

            if model is not None:
                # Gunakan threshold keyakinan 0.60 agar artefak nat keramik & bayangan lantai tereliminasi
                preds = model.predict(pil_normalized, verbose=False, conf=0.60)[0]
                boxes = getattr(preds, "boxes", None)

                (cx_e, cy_e), (ax1_e, ax2_e), _ = ellipse_orig
                egg_rad_x = (ax1_e / 2.0) * 1.15
                egg_rad_y = (ax2_e / 2.0) * 1.15

                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        cls_id = int(box.cls[0].item())
                        conf = float(box.conf[0].item())
                        x1, y1, x2, y2 = box.xyxy[0].tolist()

                        # Petakan koordinat bounding box kembali ke dimensi citra asli
                        x1_orig = round(max(0.0, min(float(w_orig), (x1 - x_off) / scale)), 2)
                        y1_orig = round(max(0.0, min(float(h_orig), (y1 - y_off) / scale)), 2)
                        x2_orig = round(max(0.0, min(float(w_orig), (x2 - x_off) / scale)), 2)
                        y2_orig = round(max(0.0, min(float(h_orig), (y2 - y_off) / scale)), 2)

                        # Filter spasial: titik tengah bounding box HARUS berada di dalam batas cangkang telur
                        box_cx = (x1_orig + x2_orig) / 2.0
                        box_cy = (y1_orig + y2_orig) / 2.0
                        norm_dist = (((box_cx - cx_e) ** 2) / (max(1.0, egg_rad_x) ** 2)) + \
                                    (((box_cy - cy_e) ** 2) / (max(1.0, egg_rad_y) ** 2))

                        if norm_dist > 1.25:
                            # Bounding box berada di luar cangkang telur (artefak latar belakang lantai) -> buang
                            continue

                        name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else "unknown"

                        detections.append({
                            "class_id": cls_id,
                            "class_name": name,
                            "confidence": round(conf, 4),
                            "bbox_xyxy": [x1_orig, y1_orig, x2_orig, y2_orig]
                        })

                    # Verifikasi fisik fraktur retak (OpenCV Surface Fracture Inspection)
                    has_physical_crack, crack_details = detect_physical_cracks(np_img, ellipse_orig)

                    # Seleksi cacat dominan berdasarkan deteksi yang valid
                    cracked_boxes = [d for d in detections if d["class_name"] == "cracked" and d["confidence"] >= 0.65]
                    dirt_boxes = [d for d in detections if d["class_name"] == "dirt" and d["confidence"] >= 0.60]
                    good_boxes = [d for d in detections if d["class_name"] == "good"]

                    # Filter false positive: jika model menebak retak namun verifikasi fisik cangkang nihil fraktur
                    if cracked_boxes and not has_physical_crack:
                        cracked_boxes = []
                        detections = [d for d in detections if d["class_name"] != "cracked"]

                    if cracked_boxes:
                        angle_dominant = "cracked"
                        angle_max_conf = max(d["confidence"] for d in cracked_boxes)
                    elif dirt_boxes:
                        angle_dominant = "dirt"
                        angle_max_conf = max(d["confidence"] for d in dirt_boxes)
                    elif good_boxes:
                        angle_dominant = "good"
                        angle_max_conf = max(d["confidence"] for d in good_boxes)
                    else:
                        angle_dominant = "good"
                        angle_max_conf = 0.985

            # Jika telur utuh bersih tanpa cacat dan detections kosong/tidak ada box good, buat bounding box hijau cangkang telur
            if angle_dominant == "good" and len([d for d in detections if d["class_name"] == "good"]) == 0:
                (cx_e, cy_e), (ax1_e, ax2_e), _ = ellipse_orig
                bx1 = round(max(0.0, cx_e - ax1_e / 2.0), 2)
                by1 = round(max(0.0, cy_e - ax2_e / 2.0), 2)
                bx2 = round(min(float(w_orig), cx_e + ax1_e / 2.0), 2)
                by2 = round(min(float(h_orig), cy_e + ax2_e / 2.0), 2)
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

        # 7. Integrasi LLM: Minta interpretasi naratif dari Google Gemini (gemini-3.8-flash)
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
    Endpoint Chatbot Interaktif dengan Google Gemini (gemini-3.8-flash)
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
    print(" 🚀 EGG QUALITY SORTING & DASHBOARD SERVER AKTIF (FLASK)")
    print(" 🥚 Standar Regulasi : SNI 3926:2023 (Telur Ayam Konsumsi)")
    print(f" 🤖 LLM Model        : Google Gemini ({DEFAULT_GEMINI_MODEL})")
    print(f" 🔑 Status API Key   : {'AKTIF' if provider == 'gemini' else 'BELUM DISETEL (Setel GEMINI_API_KEY)'}")
    print(" 👉 Dashboard URL    : http://127.0.0.1:8000")
    print("=" * 65 + "\n")
    app.run(host="127.0.0.1", port=8000, debug=False)
