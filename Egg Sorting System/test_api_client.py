"""
================================================================================
SCRIPT TEST CLIENT UNTUK FLASK API & CHATBOT GOOGLE GEMINI
Menguji:
1. Health Check (GET /api/v1/status)
2. Prediksi 360° & Grading SNI (POST /api/v1/predict)
3. Chatbot Interaktif Auditor Mutu Telur (POST /api/v1/chat)
================================================================================
"""

import os
import json
import requests

# URL Endpoint Flask
BASE_URL = "http://127.0.0.1:8000"
PREDICT_URL = f"{BASE_URL}/api/v1/predict"
CHAT_URL = f"{BASE_URL}/api/v1/chat"

# Resolusi path gambar uji secara dinamis
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_ROOT = os.path.dirname(SCRIPT_DIR)
SAMPLE_DIR = os.path.join(WORKSPACE_ROOT, "Egg Dataset", "test", "images", "cracked")

if not os.path.exists(SAMPLE_DIR):
    SAMPLE_DIR = os.path.join(WORKSPACE_ROOT, "Egg Dataset", "train", "images", "cracked")

if not os.path.exists(SAMPLE_DIR):
    print(f"[ERROR] Folder dataset tidak ditemukan di: {SAMPLE_DIR}")
    exit(1)

image_files = [f for f in os.listdir(SAMPLE_DIR) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
if not image_files:
    print(f"[ERROR] Tidak ada file citra di {SAMPLE_DIR}")
    exit(1)

test_image_path = os.path.join(SAMPLE_DIR, image_files[0])
print("=" * 65)
print(" 🧪 PENGUJIAN FLASK REST API + GEMINI CHATBOT (gemini-3.8-flash)")
print("=" * 65)
print(f"Target URL   : {BASE_URL}")
print(f"Citra Sampel : {test_image_path}")

# 1. Cek status API
try:
    status_resp = requests.get(f"{BASE_URL}/api/v1/status", timeout=5)
    print(f"\n[1] Health Check : {status_resp.status_code}")
    print(json.dumps(status_resp.json(), indent=2))
except Exception as e:
    print(f"\n[PERINGATAN] Server belum aktif di {BASE_URL}. Jalankan lebih dahulu: python app_flask.py")
    exit(1)

# 2. Kirim citra ke endpoint predict
print("\n[2] Mengirim citra untuk analisis 360° & interpretasi audit...")
inspection_data = None
with open(test_image_path, "rb") as f:
    files = {"file": (os.path.basename(test_image_path), f, "image/jpeg")}
    response = requests.post(PREDICT_URL, files=files, timeout=30)

print(f"HTTP Status: {response.status_code}")

if response.status_code == 200:
    inspection_data = response.json()
    summary = inspection_data.get("summary_360", {})
    grading = inspection_data.get("final_grading_decision", {})
    geo = inspection_data.get("opencv_geometry_inspection", {})
    llm_text = inspection_data.get("llm_interpretation", "")
    llm_meta = inspection_data.get("llm_metadata", {})

    print("\n" + "=" * 65)
    print(" 📋 RINGKASAN HASIL PREDIKSI")
    print("=" * 65)
    print(f"• Deteksi Cangkang : {summary.get('dominant_defect', '').upper()} (Confidence: {summary.get('confidence', 0)*100:.1f}%)")
    print(f"• Dimensi / Bobot  : {geo.get('length_mm')}x{geo.get('width_mm')} mm (~{geo.get('estimated_weight_gram')}g, ESI {geo.get('esi_percentage')}%)")
    print(f"• Keputusan Grade  : {grading.get('grade')} [{grading.get('grade_code')}]")
    print(f"• Aksi Konveyor    : {grading.get('hardware_sorting_action')}")
    print("=" * 65)
    print(" 🤖 INTERPRETASI AUDIT OLEH GEMINI:")
    print(f" Model: {llm_meta.get('model')} | Provider: {llm_meta.get('provider')}")
    print("-" * 65)
    print(llm_text)
    print("=" * 65)
else:
    print(f"Gagal memproses prediksi: {response.text}")

# 3. Uji Chatbot Interaktif (POST /api/v1/chat)
print("\n[3] Menguji Chatbot Interaktif Auditor Mutu Telur (POST /api/v1/chat)...")

test_questions = [
    "Kenapa telur ini tidak bisa masuk ke Mutu I?",
    "Apakah telur ini aman dikonsumsi jika dimasak?",
    "Jelaskan aturan kedalaman kantung udara pada SNI 3926:2023!"
]

for idx, q in enumerate(test_questions, start=1):
    print(f"\n--- Pertanyaan #{idx}: '{q}' ---")
    chat_payload = {
        "message": q,
        "history": [],
        "inspection_context": inspection_data
    }
    chat_resp = requests.post(CHAT_URL, json=chat_payload, timeout=20)
    if chat_resp.status_code == 200:
        cdata = chat_resp.json()
        print(f"🤖 Jawaban Auditor ({cdata.get('model', 'gemini-3.8-flash')}):")
        print(cdata.get("reply", "-"))
    else:
        print(f"Gagal chat: {chat_resp.status_code} -> {chat_resp.text}")

print("\n" + "=" * 65)
print(" ✅ PENGUJIAN SELESAI")
print("=" * 65)
