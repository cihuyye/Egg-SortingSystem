"""
================================================================================
MODUL INTEGRASI LLM (LARGE LANGUAGE MODEL) - GOOGLE GEMINI ONLY
Sistem Klasifikasi & Sortir Mutu Telur Otomatis (SNI 3926:2023)
Model: Google Gemini (gemini-3.8-flash)
================================================================================

Modul ini menghubungkan hasil inspeksi fisik cangkang (YOLOv8 360°), pengukuran
geometri/bobot (OpenCV), dan evaluasi candling ke Google Gemini API.

Fitur Unggulan:
1. Google Gemini API Only: Menggunakan model default 'gemini-3.8-flash' (dengan auto-fallback jika dibutuhkan).
2. Auto-load .env & Direct Key: Otomatis membaca GEMINI_API_KEY dari environment, file .env, atau custom API key dari client.
3. Strict Context Locking & Boundary Warning: Mengunci role LLM 100% sebagai Auditor Mutu SNI 3926:2023.
   Prompt manual pengguna divalidasi ketat. Jika di luar konteks, sistem memberikan peringatan (warning) tegas.
4. Intelligent Offline Fallback: Menjawab pertanyaan standar SNI (kantung udara, cangkang, grade) secara cerdas
   meskipun perangkat sedang offline atau API key belum disetel.
================================================================================
"""

import os
import json
import warnings
from typing import Any, Dict, List, Optional, Tuple

# Suppress deprecation warnings on generative AI SDK
warnings.filterwarnings("ignore", category=FutureWarning)

# Import Google Generative AI (Gemini)
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    genai = None
    GEMINI_AVAILABLE = False


# ==============================================================================
# KONFIGURASI MODEL & SYSTEM PROMPT AUDITOR MUTU SNI 3926:2023
# ==============================================================================
DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"

SYSTEM_PROMPT = """Kamu adalah seorang **Auditor Mutu & Ahli Standardisasi Telur Ayam Konsumsi** yang bertugas pada lini sortir otomatis berbasis **SNI 3926:2023** (Telur Ayam Konsumsi - Mutu dan Penanganan).

**PERAN & TUGAS UTAMA:**
1. Menerima data hasil inspeksi otomatis computer vision (YOLOv8 deteksi cangkang 360°, OpenCV geometri dimensi/bentuk, dan evaluasi candling).
2. Menyusun **interpretasi naratif komprehensif** dan melayani pertanyaan atau prompt manual dari pengguna / operator melalui antarmuka **Chatbot Interaktif** dalam Bahasa Indonesia.

**BATASAN KETAT & PERINGATAN (GUARDRAILS WAJIB DIPATUHI):**
1. **BATAS LINGKUP & PERINGATAN OUT-OF-CONTEXT:**
   Sistem ini HANYA melayani topik terkait:
   - Kondisi fisik kerabang/cangkang telur ayam (retak/cracked, kotoran/dirt, utuh bersih/good).
   - Pengukuran geometri dimensi (panjang, lebar), estimasi bobot, dan Egg Shape Index (ESI SNI 72–76%).
   - Uji candling (kedalaman kantung udara, posisi kuning telur, ada/tidaknya noda darah atau bercak asing).
   - Klasifikasi Grade Mutu (Mutu I / Grade A, Mutu II / Grade B, Mutu III / Reject) serta justifikasi klausul SNI 3926:2023.
   - Rekomendasi penanganan aktuator konveyor, sanitasi, dan keamanan konsumsi pangan hewani.

   JIKA PENGGUNA MEMBERIKAN PROMPT MANUAL DI LUAR TOPIK DI ATAS (misal: pemrograman/coding, matematika umum, politik, hiburan, resep masakan umum di luar telur, cerita, obrolan santai di luar topik telur, dll):
   KAMU WAJIB MEMBERIKAN PERINGATAN EKSPLISIT.
   Awali jawabanmu dengan format persis:
   "⚠️ **PERINGATAN: TOPIK DI LUAR KONTEKS!**"
   Lalu jelaskan secara tegas bahwa sistem ini dibatasi ketat hanya untuk inspeksi mutu dan standardisasi telur ayam konsumsi SNI 3926:2023.

2. **KONSISTENSI VONIS GRADE:**
   Jangan pernah membantah atau mengubah vonis Grade Akhir yang telah ditetapkan oleh aturan logika sistem (Mutu I, Mutu II, atau Mutu III). Tugasmu adalah MENJELASKAN MENGAPA keputusan tersebut diambil berdasarkan data fisik aktual dan klausul SNI.

3. **GAYA BAHASA & STRUKTUR:**
   - Gunakan Bahasa Indonesia yang baku, profesional, edukatif, dan berbasis data teknis.
   - Jika menjawab pertanyaan yang relevan: berikan jawaban yang padat, jelas, dan berbasis data.
"""

# Daftar kata kunci konteks telur untuk validasi awal prompt manual
EGG_CONTEXT_KEYWORDS = [
    "telur", "egg", "mutu", "grade", "sni", "cangkang", "kerabang", "retak", "crack",
    "kotor", "dirt", "noda", "bersih", "candling", "kantung udara", "rongga udara", "air cell",
    "berat", "bobot", "gram", "ukuran", "dimensi", "panjang", "lebar", "esi", "shape",
    "bentuk", "indeks", "lonjong", "bulat", "servo", "sortir", "konveyor", "aktuator",
    "konsumsi", "busuk", "segar", "fresh", "salmonella", "bakteri", "pangan", "haugh",
    "yolk", "kuning", "albumin", "putih", "darah", "blood", "reject", "afkir",
    "kenapa", "alasan", "mengapa", "aman", "aturan", "standar", "tabel 1", "3926",
    "ayam", "unggas", "farm", "peternakan", "cuci", "pencucian", "simpan", "penyimpanan",
    "haccp", "sanitasi", "higienis", "kualitas", "cacat", "defect", "pecah", "bocor"
]


def is_on_egg_topic(query: str) -> bool:
    """
    Memeriksa apakah prompt manual dari user berada dalam batas topik mutu telur & SNI.
    """
    q = query.lower().strip()
    for kw in EGG_CONTEXT_KEYWORDS:
        if kw in q:
            return True
    return False


def get_out_of_context_warning_response() -> Dict[str, Any]:
    """
    Menghasilkan respons peringatan terstruktur jika prompt manual di luar konteks.
    """
    warning_text = (
        "⚠️ **PERINGATAN: TOPIK DI LUAR KONTEKS SISTEM!**\n\n"
        "Prompt atau pertanyaan Anda terdeteksi tidak relevan dengan sistem sortir mutu telur konsumsi.\n\n"
        "**Batasan Topik yang Diperbolehkan:**\n"
        "• 🔍 **Cacat Cangkang**: Evaluasi retak (cracked), noda kotoran (dirt), dan kebersihan cangkang 360°\n"
        "• 📐 **Geometri & Bobot**: Dimensi panjang/lebar, berat gram, dan Egg Shape Index (ESI ideal 72–76%)\n"
        "• 💡 **Uji Candling**: Kedalaman kantung udara, posisi kuning telur, dan bercak darah\n"
        "• 📋 **Regulasi SNI**: Klausul SNI 3926:2023, justifikasi penetapan Mutu I, II, atau III (Reject)\n"
        "• ⚙️ **Operasional**: Rekomendasi aktuator servo jalur konveyor dan keamanan pangan konsumsi\n\n"
        "*Silakan masukkan instruksi manual atau pertanyaan yang berkaitan dengan parameter di atas.*"
    )
    return {
        "status": "warning",
        "is_warning": True,
        "reply": warning_text,
        "model": "Auditor Guardrail Engine",
        "provider": "Context Validator"
    }


def load_dotenv_key() -> str:
    """
    Mencari dan membaca GEMINI_API_KEY dari file .env secara otomatis.
    """
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(cur_dir, "..", ".env"),
        os.path.join(cur_dir, "..", "..", ".env"),
        os.path.join(os.getcwd(), ".env"),
        os.path.join(os.getcwd(), "Egg Sorting System", ".env")
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("GEMINI_API_KEY=") and not line.startswith("#"):
                            val = line.split("=", 1)[1].strip().strip('"').strip("'")
                            if val:
                                return val
            except Exception:
                pass
    return ""


def detect_active_provider(custom_api_key: Optional[str] = None) -> Tuple[str, str]:
    """
    Mendeteksi apakah Gemini API aktif.
    Memeriksa parameter custom_api_key, environment GEMINI_API_KEY, atau file .env.
    """
    if custom_api_key and custom_api_key.strip():
        return "gemini", custom_api_key.strip()

    gemini_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not gemini_key:
        gemini_key = load_dotenv_key()
        if gemini_key:
            os.environ["GEMINI_API_KEY"] = gemini_key

    if gemini_key:
        return "gemini", gemini_key
    return "none", ""


def build_inspection_prompt(inspection_data: Dict[str, Any]) -> str:
    """
    Menyusun prompt data inspeksi yang terstruktur rapi untuk dianalisis LLM.
    """
    summary = inspection_data.get("summary_360", {})
    geometry = inspection_data.get("opencv_geometry_inspection", {})
    grading = inspection_data.get("final_grading_decision", {})
    total_angles = inspection_data.get("total_angles", 1)

    prompt = f"""Tolong susun interpretasi naratif audit mutu telur berikut berdasarkan standar SNI 3926:2023:

[DATA HASIL INSPEKSI COMPUTER VISION]
1. CANGKANG LUAR (YOLOv8 360° Inspection):
   - Total Sudut Diperiksa: {total_angles} sudut pandang
   - Kesimpulan Dominan: {summary.get('dominant_defect', 'good').upper()}
   - Keyakinan Deteksi: {summary.get('confidence', 1.0) * 100:.1f}%
   - Cacat Retak: {'DITEMUKAN RETAK' if summary.get('has_crack') else 'Bebas Retak (Utuh)'}
   - Cacat Kotoran: {'DITEMUKAN NODA KOTORAN' if summary.get('has_dirt') else 'Bersih'}
   - Rincian Sudut: {summary.get('notes', '-')}

2. GEOMETRI & BOBOT (OpenCV Measurement):
   - Dimensi: Panjang {geometry.get('length_mm', 0)} mm, Lebar {geometry.get('width_mm', 0)} mm
   - Egg Shape Index (ESI): {geometry.get('esi_percentage', 0)}% (Status: {geometry.get('shape_status', 'Normal')})
   - Estimasi Bobot: {geometry.get('estimated_weight_gram', 0)} gram
   - Kategori Ukuran SNI: {geometry.get('size_category', 'Sedang')}

3. KEPUTUSAN KUALITAS AKHIR (SNI 3926:2023 Decision Engine):
   - Grade Mutu: {grading.get('grade', 'Mutu I (Grade A)')}
   - Kode Grade: {grading.get('grade_code', 'MUTU_I')}
   - Faktor Utama: {grading.get('yolo_primary_factor', '-')}
   - Alasan Sistem: {grading.get('reason', '-')}
   - Aksi Hardware Servo: {grading.get('hardware_sorting_action', '-')}
   - Acuan Regulasi: {grading.get('standard_reference', 'SNI 3926:2023')}

Susun interpretasi profesional, lugas, dan terstruktur sesuai instruksi perananmu."""

    return prompt


def generate_with_gemini(
    prompt: str,
    api_key: str,
    model_name: str = DEFAULT_GEMINI_MODEL
) -> str:
    """
    Memanggil Google Gemini API untuk menghasilkan narasi interpretasi.
    Model default: gemini-3.8-flash (dengan fallback alias otomatis jika perlu).
    """
    if not GEMINI_AVAILABLE:
        return "Library 'google-generativeai' belum terpasang. Jalankan: pip install google-generativeai"

    model_candidates = [model_name, "gemini-2.0-flash", "gemini-1.5-flash"]
    last_err = ""

    for candidate in model_candidates:
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(
                model_name=candidate,
                system_instruction=SYSTEM_PROMPT
            )
            response = model.generate_content(prompt)
            if response and response.text:
                return response.text.strip()
        except Exception as e:
            last_err = str(e)
            continue

    return f"Gagal memanggil Gemini API ({model_name}): {last_err}"


def get_llm_interpretation(
    inspection_data: Dict[str, Any],
    custom_api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Fungsi utama untuk menghasilkan interpretasi naratif awal dari data inspeksi telur.
    Mengembalikan dict dengan teks interpretasi, model Gemini, dan status.
    """
    provider, api_key = detect_active_provider(custom_api_key)
    prompt = build_inspection_prompt(inspection_data)

    if provider == "gemini":
        narrative = generate_with_gemini(prompt, api_key, model_name=DEFAULT_GEMINI_MODEL)
        return {
            "status": "success",
            "provider": "Google Gemini",
            "model": DEFAULT_GEMINI_MODEL,
            "role": "Auditor Mutu SNI 3926:2023",
            "text": narrative
        }

    # Fallback offline terstruktur jika GEMINI_API_KEY belum disetel
    grading = inspection_data.get("final_grading_decision", {})
    grade = grading.get("grade", "Mutu I (Grade A)")
    reason = grading.get("reason", "Cangkang utuh dan bersih")
    action = grading.get("hardware_sorting_action", "Kemas ke wadah standar")
    shape = inspection_data.get("opencv_geometry_inspection", {})
    esi = shape.get("esi_percentage", 74.0)
    weight = shape.get("estimated_weight_gram", 55.0)
    size_cat = shape.get("size_category", "Sedang")

    offline_text = (
        f"Telur ini terklasifikasi sebagai **{grade}** berdasarkan standar SNI 3926:2023. "
        f"Alasan utama penetapan adalah: {reason}. "
        f"Dari sisi fisik, telur memiliki estimasi bobot {weight:.1f} gram (kategori {size_cat}) "
        f"dengan Shape Index {esi:.1f}%.\n\n"
        f"Rekomendasi tindakan operasional: {action}."
    )

    return {
        "status": "offline_fallback",
        "provider": "Rule-based Engine (Offline)",
        "model": "None (Setel GEMINI_API_KEY)",
        "role": "Auditor Mutu SNI 3926:2023",
        "text": offline_text
    }


def chat_with_gemini(
    user_message: str,
    history: Optional[List[Dict[str, str]]] = None,
    inspection_context: Optional[Dict[str, Any]] = None,
    custom_api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Fungsi Chatbot interaktif: Menjawab pertanyaan atau prompt manual pengguna
    dengan batasan ketat konteks mutu telur SNI 3926:2023.
    Jika di luar batas, menghasilkan peringatan (warning).
    """
    # 1. Validasi awal batasan topik (Guardrail layer 1)
    if not is_on_egg_topic(user_message) and len(user_message.split()) > 1:
        return get_out_of_context_warning_response()

    provider, api_key = detect_active_provider(custom_api_key)

    # Susun konteks inspeksi telur jika tersedia
    context_str = ""
    if inspection_context:
        summary = inspection_context.get("summary_360", {})
        grading = inspection_context.get("final_grading_decision", {})
        geo = inspection_context.get("opencv_geometry_inspection", {})
        context_str = (
            f"[DATA TELUR SAAT INI YANG SEDANG DIINSPEKSI]\n"
            f"- Vonis Grade Akhir: {grading.get('grade', 'Belum dianalisis')}\n"
            f"- Cacat Cangkang (YOLO 360°): {summary.get('dominant_defect', 'unknown').upper()} "
            f"(Confidence: {summary.get('confidence', 1.0) * 100:.1f}%, Catatan: {summary.get('notes', '-')})\n"
            f"- Dimensi & Bentuk (OpenCV): {geo.get('length_mm', 0)}x{geo.get('width_mm', 0)} mm, "
            f"Shape Index (ESI): {geo.get('esi_percentage', 0)}% ({geo.get('shape_status', '-')})\n"
            f"- Bobot & Kategori SNI: ~{geo.get('estimated_weight_gram', 0)}g ({geo.get('size_category', '-')})\n"
            f"- Alasan Penetapan: {grading.get('reason', '-')}\n"
            f"- Aksi Konveyor: {grading.get('hardware_sorting_action', '-')}\n"
            f"--------------------------------------------------\n"
        )

    # 2. Mode Online dengan Gemini API (Aktif jika API key tersedia)
    if provider == "gemini" and GEMINI_AVAILABLE:
        model_candidates = [DEFAULT_GEMINI_MODEL, "gemini-2.0-flash", "gemini-1.5-flash"]
        last_error = ""

        gemini_contents = []

        if context_str:
            gemini_contents.append({
                "role": "user",
                "parts": [f"Berikut data telur yang baru saja selesai diinspeksi:\n{context_str}\nPahami data di atas untuk menjawab pertanyaan-pertanyaan berikutnya."]
            })
            gemini_contents.append({
                "role": "model",
                "parts": ["Baik, data hasil inspeksi telur telah saya catat dan pahami sesuai acuan SNI 3926:2023. Silakan ajukan pertanyaan terkait telur ini atau standardisasi mutu telur ayam konsumsi."]
            })

        if history:
            for turn in history:
                role = "user" if turn.get("role") in ["user", "human"] else "model"
                content = turn.get("content", "").strip()
                if content:
                    gemini_contents.append({
                        "role": role,
                        "parts": [content]
                    })

        gemini_contents.append({
            "role": "user",
            "parts": [user_message]
        })

        for candidate in model_candidates:
            try:
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel(
                    model_name=candidate,
                    system_instruction=SYSTEM_PROMPT
                )
                response = model.generate_content(gemini_contents)
                if response and response.text:
                    reply_text = response.text.strip()
                    is_warn = "PERINGATAN: TOPIK DI LUAR KONTEKS" in reply_text or "⚠️" in reply_text[:30]
                    return {
                        "status": "warning" if is_warn else "success",
                        "is_warning": is_warn,
                        "reply": reply_text,
                        "model": candidate,
                        "provider": "Google Gemini"
                    }
            except Exception as e:
                last_error = str(e)
                continue

        return {
            "status": "error",
            "is_warning": False,
            "reply": f"Maaf, gagal memanggil Gemini API ({DEFAULT_GEMINI_MODEL}): {last_error}",
            "model": DEFAULT_GEMINI_MODEL,
            "provider": "Google Gemini"
        }

    # 3. Mode Offline Fallback Cerdas (SNI Knowledge Engine)
    # Jika API key belum disetel, jawab pertanyaan standar SNI dengan akurat
    q_lower = user_message.lower()

    # A. Pertanyaan khusus Kantung Udara / Rongga Udara / Air Cell
    if "rongga udara" in q_lower or "kantung udara" in q_lower or "air cell" in q_lower:
        reply = (
            "Berdasarkan standar **SNI 3926:2023 Tabel 1** (Syarat Mutu Fisik Telur Konsumsi):\n\n"
            "• **Mutu I (Grade A)**: Kedalaman kantung udara **< 0,5 cm (< 5 mm)** dan posisinya tetap (*fixed*) stabil di ujung tumpul telur.\n"
            "• **Mutu II (Grade B)**: Kedalaman kantung udara **0,5 cm – 0,9 cm (5 – 9 mm)** dan posisinya boleh sedikit bergerak bebas.\n"
            "• **Mutu III (Reject)**: Kedalaman kantung udara **> 0,9 cm (> 9 mm)** atau kantung udara bebas bergerak/pecah.\n\n"
            "*(Semakin dalam kantung udara, menandakan telur sudah disimpan lama dan kesegarannya berkurang akibat penguapan)*"
        )
        return {
            "status": "offline_fallback",
            "is_warning": False,
            "reply": reply,
            "model": "Rule-based Engine (Offline)",
            "provider": "Local Fallback"
        }

    # B. Pertanyaan spesifikasi Mutu III / Reject (Cek terlebih dahulu sebelum Mutu II / Mutu I)
    if "mutu iii" in q_lower or "mutu 3" in q_lower or "grade c" in q_lower or "reject" in q_lower or "afkir" in q_lower:
        reply = (
            "Kriteria **Mutu III (Reject / Afkir)** menurut **SNI 3926:2023**:\n"
            "1. **Kerabang/Cangkang**: Retak (*cracked*), bocor, atau berlumur kotoran parah.\n"
            "2. **Kantung Udara**: Kedalaman > 0,9 cm atau kantung udara pecah/bebas bergerak.\n"
            "3. **Candling**: Terdapat noda darah (*blood spots*), embrio, atau bau busuk.\n\n"
            "Telur dengan kriteria ini dialihkan oleh aktuator konveyor ke wadah reject untuk mencegah cemaran bakteri."
        )
        return {
            "status": "offline_fallback",
            "is_warning": False,
            "reply": reply,
            "model": "Rule-based Engine (Offline)",
            "provider": "Local Fallback"
        }

    # C. Pertanyaan spesifikasi Mutu II
    if "mutu ii" in q_lower or "mutu 2" in q_lower or "grade b" in q_lower:
        reply = (
            "Kriteria **Mutu II (Grade B)** menurut **SNI 3926:2023**:\n"
            "1. **Kerabang/Cangkang**: Utuh, boleh terdapat sedikit noda kotoran (< 1/16 luas permukaan).\n"
            "2. **Indeks Bentuk (ESI)**: Boleh sedikit di luar 72–76% (agak lonjong atau agak bulat).\n"
            "3. **Kantung Udara**: Kedalaman 0,5 cm – 0,9 cm.\n"
            "4. **Kondisi Dalam**: Bebas noda darah dan benda asing."
        )
        return {
            "status": "offline_fallback",
            "is_warning": False,
            "reply": reply,
            "model": "Rule-based Engine (Offline)",
            "provider": "Local Fallback"
        }

    # D. Pertanyaan spesifikasi Mutu I
    if "mutu i" in q_lower or "mutu 1" in q_lower or "grade a" in q_lower:
        reply = (
            "Kriteria **Mutu I (Grade A)** menurut **SNI 3926:2023**:\n"
            "1. **Kerabang/Cangkang**: Utuh, bersih dari kotoran/feses, tekstur halus normal.\n"
            "2. **Indeks Bentuk (ESI)**: 72% – 76% (proporsional normal).\n"
            "3. **Kantung Udara**: Kedalaman < 0,5 cm, posisi tetap tidak berpindah.\n"
            "4. **Kondisi Dalam**: Bebas dari noda darah (*blood spots*) maupun benda asing."
        )
        return {
            "status": "offline_fallback",
            "is_warning": False,
            "reply": reply,
            "model": "Rule-based Engine (Offline)",
            "provider": "Local Fallback"
        }

    # E. Pertanyaan terkait telur aktif yang baru diperiksa
    if inspection_context:
        grading = inspection_context.get("final_grading_decision", {})
        grade = grading.get("grade", "Mutu I (Grade A)")
        reason = grading.get("reason", "Cangkang utuh dan bersih")
        action = grading.get("hardware_sorting_action", "Kemas ke wadah standar")
        geo = inspection_context.get("opencv_geometry_inspection", {})
        summary = inspection_context.get("summary_360", {})

        if "kenapa" in q_lower or "mengapa" in q_lower or "alasan" in q_lower or "grade" in q_lower:
            reply = (
                f"Telur ini diputuskan masuk ke dalam **{grade}** karena: {reason}. "
                f"Berdasarkan SNI 3926:2023 Tabel 1, telur dengan kondisi cangkang "
                f"'{summary.get('dominant_defect', 'good')}' dan indeks bentuk {geo.get('esi_percentage', 74)}% "
                f"harus diarahkan ke: {action}."
            )
        elif "aman" in q_lower or "konsumsi" in q_lower or "makan" in q_lower:
            if "III" in grade or "reject" in grade.lower() or summary.get("has_crack"):
                reply = (
                    "⚠️ **Peringatan Keamanan Pangan**: Telur dengan cangkang retak (Mutu III) memiliki risiko tinggi "
                    "terkontaminasi bakteri patogen (seperti *Salmonella sp.*) melalui celah kerabang yang pecah. "
                    "Telur ini tidak disarankan untuk dijual segar dan harus dipisahkan dari peredaran konsumsi meja."
                )
            elif "II" in grade:
                reply = (
                    "✅ **Aman Dikonsumsi Bersyarat**: Telur Mutu II umumnya aman dikonsumsi jika cangkangnya masih utuh "
                    "meskipun terdapat sedikit noda kotoran atau bentuknya sedikit lonjong/bulat. Bersihkan cangkang "
                    "dan masak secara matang sebelum dikonsumsi."
                )
            else:
                reply = (
                    "✅ **Sangat Aman & Segar**: Telur Mutu I memenuhi seluruh kriteria mutu kerabang utuh, bersih, "
                    "dan segar menurut SNI 3926:2023. Sangat layak dipasarkan sebagai telur konsumsi grade premium."
                )
        elif "bentuk" in q_lower or "esi" in q_lower or "ukuran" in q_lower or "bobot" in q_lower or "berat" in q_lower:
            reply = (
                f"Berdasarkan pengukuran OpenCV, telur ini memiliki dimensi {geo.get('length_mm')}x{geo.get('width_mm')} mm, "
                f"Egg Shape Index (ESI) {geo.get('esi_percentage')}%, dan estimasi bobot ~{geo.get('estimated_weight_gram')}g "
                f"(Kategori SNI: {geo.get('size_category')}). "
                f"Standar ESI ideal SNI 3926:2023 adalah 72% – 76%."
            )
        elif "rekomendasi" in q_lower or "tindakan" in q_lower or "servo" in q_lower:
            reply = (
                f"Rekomendasi tindakan operasional untuk telur ini: **{action}**. "
                f"Sistem aktuator conveyor akan mengaktifkan pintu sortir sesuai kode grade {grading.get('grade_code')}."
            )
        else:
            reply = (
                f"Hasil inspeksi telur aktif: **{grade}**. Kondisi cangkang: {summary.get('dominant_defect', 'good').upper()} "
                f"(Catatan: {summary.get('notes', '-')}). {reason}. "
                f"Rekomendasi penanganan: {action}."
            )
    else:
        reply = (
            "Halo! Saya adalah **Auditor Mutu AI Telur SNI 3926:2023**. "
            "Silakan unggah foto telur pada dashboard dan klik **⚡ Analisis 360°**, "
            "maka saya akan mengaudit dan menjawab prompt manual atau pertanyaan spesifik mengenai mutu telur tersebut."
        )

    return {
        "status": "offline_fallback",
        "is_warning": False,
        "reply": reply,
        "model": "Rule-based Engine (Offline)",
        "provider": "Local Fallback"
    }
