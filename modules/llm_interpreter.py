"""
================================================================================
MODUL INTEGRASI LLM (LARGE LANGUAGE MODEL) - GOOGLE GEMINI ONLY
Sistem Klasifikasi & Sortir Mutu Telur Otomatis (SNI 3926:2023)
Model: Google Gemini (gemini-3.8-flash)
SDK: google-genai (resmi, menggantikan google-generativeai yang deprecated)
================================================================================

Modul ini menghubungkan hasil inspeksi fisik cangkang (YOLOv8 360°) dan pengukuran
geometri/bobot (OpenCV) ke Google Gemini API.

Fitur Unggulan:
1. Google Gemini API Only: Menggunakan model default 'gemini-3.8-flash' (dengan auto-fallback jika dibutuhkan).
2. SDK Resmi Terbaru: Menggunakan `google.genai` (bukan `google.generativeai` yang deprecated).
3. Auto-load .env & Direct Key: Otomatis membaca GEMINI_API_KEY dari environment, file .env, atau custom API key dari client.
4. Strict Context Locking & Boundary Warning: Mengunci role LLM 100% sebagai Auditor Mutu SNI 3926:2023.
   Prompt manual pengguna divalidasi ketat. Jika di luar konteks, sistem memberikan peringatan (warning) tegas.
5. Intelligent Offline Fallback: Menjawab pertanyaan standar SNI (cacat cangkang, bentuk ESI, grade mutu) secara cerdas
   meskipun perangkat sedang offline atau API key belum disetel.
6. Anti-Hallucination Engine: temperature rendah (0.2), grounding instruction eksplisit,
   output validation, negative keyword filter, dan history limiter.
================================================================================
"""

import os
import json
import re
import warnings
from typing import Any, Dict, List, Optional, Tuple

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# ==============================================================================
# IMPORT GOOGLE GENAI SDK (RESMI — BUKAN DEPRECATED)
# ==============================================================================
try:
    from google import genai
    from google.genai import types as genai_types
    GEMINI_AVAILABLE = True
except ImportError:
    genai = None
    genai_types = None
    GEMINI_AVAILABLE = False


# ==============================================================================
# KONFIGURASI MODEL & GENERATION PARAMETERS (ANTI-HALLUCINATION)
# ==============================================================================
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash"

# [OPTIMASI #1] Temperature rendah + token limit untuk menekan halusinasi
GENERATION_CONFIG = {
    "temperature": 0.2,          # Sangat deterministik, berbasis fakta
    "max_output_tokens": 1024,   # Cegah over-generation (sumber halusinasi)
    "top_p": 0.8,                # Fokus pada token paling probable
    "top_k": 40,                 # Batasi sampling vocabulary
}

# [OPTIMASI #4] Batas maksimal chat history untuk menjaga context window
MAX_CHAT_HISTORY_TURNS = 10


# ==============================================================================
# SYSTEM PROMPT AUDITOR MUTU SNI 3926:2023 (DIPERKUAT ANTI-HALUSINASI)
# ==============================================================================
# [OPTIMASI #7] System prompt diperkuat dengan aturan anti-halusinasi eksplisit
SYSTEM_PROMPT = """Kamu adalah seorang **Auditor Mutu & Ahli Standardisasi Telur Ayam Konsumsi** yang bertugas pada lini sortir otomatis berbasis **SNI 3926:2023** (Telur Ayam Konsumsi - Mutu dan Penanganan).

**PERAN & TUGAS UTAMA:**
1. Menerima data hasil inspeksi otomatis computer vision (YOLOv8 deteksi cangkang 360° dan OpenCV geometri dimensi/bentuk/bobot).
2. Menyusun **interpretasi naratif komprehensif** dan melayani pertanyaan atau prompt manual dari pengguna / operator melalui antarmuka **Chatbot Interaktif** dalam Bahasa Indonesia.

**ATURAN ANTI-HALUSINASI (WAJIB DIPATUHI MUTLAK):**
1. **GROUNDING KETAT**: HANYA gunakan data dan angka yang secara eksplisit tersedia di dalam data inspeksi yang diberikan. DILARANG KERAS mengarang, memperkirakan, atau menambahkan angka, persentase, ukuran, bobot, atau fakta apa pun yang TIDAK ADA di dalam data input.
2. **KONSISTENSI VONIS GRADE**: JANGAN PERNAH membantah, mengubah, meragukan, atau menyarankan grade berbeda dari Grade Akhir yang telah ditetapkan oleh Decision Engine sistem. Tugasmu HANYA menjelaskan MENGAPA keputusan tersebut diambil berdasarkan data yang diberikan dan klausul SNI 3926:2023.
3. **JANGAN MENGARANG KLAUSUL SNI**: Hanya kutip klausul dan ketentuan SNI 3926:2023 yang kamu ketahui dengan pasti. Jika tidak yakin, cukup sebut "berdasarkan ketentuan SNI 3926:2023" tanpa menyebut nomor pasal/klausul spesifik yang belum pasti.
4. **JANGAN MENAMBAH REKOMENDASI MEDIS/KLINIS**: Jangan memberikan saran medis, diagnosis penyakit unggas, atau rekomendasi pengobatan. Batasi pada mutu cangkang, sortir, dan keamanan pangan dasar.
5. **JIKA RAGU, KATAKAN TIDAK TAHU**: Jika data tidak cukup untuk menjawab, katakan "data yang tersedia tidak mencukupi untuk menjawab pertanyaan tersebut" daripada mengarang jawaban.

**BATASAN KETAT & PERINGATAN (GUARDRAILS WAJIB DIPATUHI):**
1. **BATAS LINGKUP & PERINGATAN OUT-OF-CONTEXT:**
   Sistem ini HANYA melayani topik terkait:
   - Kondisi fisik kerabang/cangkang telur ayam (retak/cracked, kotoran/dirt, utuh bersih/good).
   - Pengukuran geometri dimensi (panjang, lebar), estimasi bobot, dan Egg Shape Index (ESI SNI 72–76%).
   - Klasifikasi Grade Mutu (Mutu I / Grade A, Mutu II / Grade B, Mutu III / Reject) serta justifikasi klausul SNI 3926:2023.
   - Rekomendasi penanganan aktuator konveyor, sanitasi, dan keamanan konsumsi pangan hewani.

   JIKA PENGGUNA MEMBERIKAN PROMPT MANUAL DI LUAR TOPIK DI ATAS (misal: pemrograman/coding, matematika umum, politik, hiburan, resep masakan umum di luar telur, cerita, obrolan santai di luar topik telur, dll):
   KAMU WAJIB MEMBERIKAN PERINGATAN EKSPLISIT.
   Awali jawabanmu dengan format persis:
   "⚠️ **PERINGATAN: TOPIK DI LUAR KONTEKS!**"
   Lalu jelaskan secara tegas bahwa sistem ini dibatasi ketat hanya untuk inspeksi mutu dan standardisasi telur ayam konsumsi SNI 3926:2023.

2. **GAYA BAHASA & STRUKTUR:**
   - Gunakan Bahasa Indonesia yang baku, profesional, edukatif, dan berbasis data teknis.
   - Jika menjawab pertanyaan yang relevan: berikan jawaban yang padat, jelas, dan berbasis data.
   - Jangan menambahkan data atau angka yang tidak ada di input inspeksi.
"""


# ==============================================================================
# KEYWORD FILTER — TOPIC VALIDATION (DIPERKUAT)
# ==============================================================================
# Daftar kata kunci konteks telur untuk validasi awal prompt manual
EGG_CONTEXT_KEYWORDS = [
    "telur", "egg", "mutu", "grade", "sni", "cangkang", "kerabang", "retak", "crack",
    "kotor", "dirt", "noda", "bersih", "feses", "bintik",
    "berat", "bobot", "gram", "ukuran", "dimensi", "panjang", "lebar", "esi", "shape",
    "bentuk", "indeks", "lonjong", "bulat", "servo", "sortir", "konveyor", "aktuator",
    "konsumsi", "busuk", "segar", "fresh", "salmonella", "bakteri", "pangan", "haugh",
    "reject", "afkir", "kenapa", "alasan", "mengapa", "aman", "aturan", "standar", "tabel 1", "3926",
    "ayam", "unggas", "farm", "peternakan", "cuci", "pencucian", "simpan", "penyimpanan",
    "haccp", "sanitasi", "higienis", "kualitas", "cacat", "defect", "pecah", "bocor"
]

# [OPTIMASI #3] Negative keywords — topik yang PASTI di luar konteks
OFF_TOPIC_KEYWORDS = [
    "coding", "programming", "python", "javascript", "html", "css", "react", "flutter",
    "code", "script", "function", "variable", "class ", "import ",
    "politik", "pemilu", "presiden", "partai", "pilkada",
    "cuaca", "weather", "zodiak", "horoscope", "ramalan",
    "game", "gaming", "minecraft", "fortnite", "valorant", "mobile legend",
    "film", "movie", "drama", "anime", "manga", "netflix", "youtube",
    "lagu", "musik", "song", "lirik", "spotify",
    "dongeng", "cerita fiksi", "novel", "puisi", "pantun",
    "matematika", "kalkulus", "integral", "persamaan", "logaritma",
    "translate", "terjemahkan", "bahasa inggris",
    "bitcoin", "crypto", "saham", "trading", "forex",
    "resep kue", "resep masak", "cara membuat", "bahan bahan",
    "obat", "penyakit", "diagnos", "gejala", "terapi",
    "hack", "cheat", "bypass", "crack ", "keygen",
    "asmara", "jodoh", "pacar", "cinta",
]


def is_on_egg_topic(query: str) -> bool:
    """
    [OPTIMASI #3] Memeriksa apakah prompt manual dari user berada dalam batas
    topik mutu telur & SNI. Diperkuat dengan negative keyword blacklist.
    """
    q = query.lower().strip()

    # Cek negative keywords dulu — jika ada, langsung tolak
    for neg in OFF_TOPIC_KEYWORDS:
        if neg in q:
            return False

    # Cek positive keywords
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


# ==============================================================================
# GROUNDING INSTRUCTION — ANTI-FABRICATION PROMPT LAYER
# ==============================================================================
# [OPTIMASI #2] Instruksi grounding eksplisit yang disisipkan di setiap prompt data
GROUNDING_INSTRUCTION = """
⚠️ INSTRUKSI GROUNDING (WAJIB):
- Gunakan HANYA data dan angka yang tertera di atas. JANGAN menambahkan angka, persentase, dimensi, bobot, atau fakta lain yang TIDAK ADA di data inspeksi.
- JANGAN mengubah atau meragukan Grade Akhir yang ditetapkan oleh Decision Engine.
- Jika ada data yang tidak tersedia atau bernilai 0, katakan "data tidak tersedia" daripada mengarang nilai.
- Jangan menyebut nomor klausul/pasal SNI spesifik kecuali kamu yakin 100% akurat.
"""


def build_inspection_prompt(inspection_data: Dict[str, Any]) -> str:
    """
    Menyusun prompt data inspeksi yang terstruktur rapi untuk dianalisis LLM.
    [OPTIMASI #2] Dilengkapi grounding instruction eksplisit.
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
{GROUNDING_INSTRUCTION}
Susun interpretasi profesional, lugas, dan terstruktur sesuai instruksi perananmu."""

    return prompt


# ==============================================================================
# POST-PROCESSING — GRADE CONSISTENCY VALIDATION
# ==============================================================================
# [OPTIMASI #5] Validasi output LLM: pastikan tidak membantah grade sistem
def validate_grade_consistency(llm_output: str, actual_grade: str) -> Tuple[str, bool]:
    """
    Memeriksa apakah output LLM konsisten dengan grade yang ditetapkan sistem.
    Jika LLM menyebut grade berbeda, tambahkan disclaimer.

    Returns:
        Tuple[str, bool]: (output yang sudah divalidasi, apakah ada inkonsistensi)
    """
    if not actual_grade or not llm_output:
        return llm_output, False

    # Tentukan grade code yang seharusnya
    grade_lower = actual_grade.lower()
    if "mutu i" in grade_lower or "grade a" in grade_lower:
        conflicting = ["mutu ii", "mutu iii", "grade b", "grade c", "reject", "mutu 2", "mutu 3"]
    elif "mutu ii" in grade_lower or "grade b" in grade_lower:
        conflicting = ["mutu i ", "mutu iii", "grade a", "grade c", "reject", "mutu 1 ", "mutu 3"]
    elif "mutu iii" in grade_lower or "reject" in grade_lower or "grade c" in grade_lower:
        conflicting = ["mutu i ", "mutu ii", "grade a", "grade b", "mutu 1 ", "mutu 2"]
    else:
        return llm_output, False

    output_lower = llm_output.lower()

    # Cek apakah LLM menyebutkan grade yang berlawanan
    has_conflict = False
    for conf in conflicting:
        # Cari pola di mana LLM menyarankan grade berbeda
        conflict_patterns = [
            f"seharusnya {conf}",
            f"sebaiknya {conf}",
            f"lebih tepat {conf}",
            f"sebenarnya {conf}",
            f"koreksi menjadi {conf}",
            f"diubah menjadi {conf}",
        ]
        for pattern in conflict_patterns:
            if pattern in output_lower:
                has_conflict = True
                break
        if has_conflict:
            break

    if has_conflict:
        disclaimer = (
            f"\n\n---\n"
            f"📌 **Catatan Sistem**: Grade akhir telur ini telah ditetapkan sebagai "
            f"**{actual_grade}** oleh Decision Engine SNI 3926:2023 berdasarkan "
            f"parameter terukur. Keputusan grade bersifat final dan tidak dapat "
            f"diubah oleh interpretasi naratif."
        )
        return llm_output + disclaimer, True

    return llm_output, False


# [OPTIMASI #6] Safety post-filter — cek rekomendasi berbahaya
def safety_post_filter(llm_output: str, inspection_data: Optional[Dict] = None) -> str:
    """
    Memeriksa apakah output LLM mengandung rekomendasi yang berpotensi berbahaya
    untuk keamanan pangan, dan menambahkan disclaimer jika perlu.
    """
    if not llm_output or not inspection_data:
        return llm_output

    output_lower = llm_output.lower()
    summary = inspection_data.get("summary_360", {})
    has_crack = summary.get("has_crack", False)

    # Jika telur retak tapi LLM bilang aman dimakan tanpa syarat
    dangerous_phrases = [
        "aman dikonsumsi langsung",
        "aman dimakan mentah",
        "aman untuk dikonsumsi segar",
        "layak konsumsi segar",
    ]

    if has_crack:
        for phrase in dangerous_phrases:
            if phrase in output_lower:
                safety_note = (
                    "\n\n⚠️ **Peringatan Keamanan Pangan**: Telur dengan cangkang retak "
                    "memiliki risiko tinggi kontaminasi bakteri patogen (*Salmonella sp.*). "
                    "Telur ini TIDAK boleh dikonsumsi mentah/setengah matang dan harus "
                    "dimasak hingga matang sempurna atau dipisahkan dari peredaran."
                )
                return llm_output + safety_note

    return llm_output


# ==============================================================================
# GEMINI CLIENT FACTORY — Membuat client google.genai
# ==============================================================================
def _create_gemini_client(api_key: str) -> Any:
    """
    Membuat instance google.genai.Client dengan API key dan konfigurasi v1beta.
    """
    if genai is None or genai_types is None:
        return None
    return genai.Client(
        api_key=api_key,
        http_options=genai_types.HttpOptions(api_version="v1beta")
    )


def _build_gen_config(extra: Optional[Dict] = None) -> Any:
    """
    Membuat GenerateContentConfig dari GENERATION_CONFIG + system prompt.
    """
    if genai_types is None:
        return None
    params = {
        "system_instruction": SYSTEM_PROMPT,
        "temperature": GENERATION_CONFIG["temperature"],
        "max_output_tokens": GENERATION_CONFIG["max_output_tokens"],
        "top_p": GENERATION_CONFIG["top_p"],
        "top_k": GENERATION_CONFIG["top_k"],
    }
    if extra:
        params.update(extra)
    return genai_types.GenerateContentConfig(**params)


# ==============================================================================
# GEMINI API CALLER (SDK BARU google.genai)
# ==============================================================================
def generate_with_gemini(
    prompt: str,
    api_key: str,
    model_name: str = DEFAULT_GEMINI_MODEL
) -> str:
    """
    Memanggil Google Gemini API untuk menghasilkan narasi interpretasi.
    Menggunakan SDK baru `google.genai` (bukan deprecated `google.generativeai`).
    [OPTIMASI #1] Menggunakan temperature=0.2 dan max_output_tokens=1024.
    """
    if genai is None or genai_types is None:
        return "Library 'google-genai' belum terpasang. Jalankan: pip install google-genai"

    model_candidates = [
        model_name,
        "gemini-3.5-flash",
        "gemini-3.5-flash-lite",
        "gemini-3.6-flash",
        "gemini-3-flash-preview",
        "gemini-3.8-flash",
        "gemini-3.7-flash"
    ]
    model_candidates = list(dict.fromkeys(model_candidates))
    last_err = ""

    client = _create_gemini_client(api_key)
    if client is None:
        return "Gagal membuat Gemini client."

    config = _build_gen_config()

    for candidate in model_candidates:
        try:
            response = client.models.generate_content(
                model=candidate,
                contents=prompt,
                config=config,
            )
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
    [OPTIMASI #2, #5, #6] Dengan grounding, grade validation, dan safety filter.
    """
    provider, api_key = detect_active_provider(custom_api_key)
    prompt = build_inspection_prompt(inspection_data)

    if provider == "gemini":
        narrative = generate_with_gemini(prompt, api_key, model_name=DEFAULT_GEMINI_MODEL)

        # [OPTIMASI #5] Validasi konsistensi grade
        actual_grade = inspection_data.get("final_grading_decision", {}).get("grade", "")
        narrative, had_conflict = validate_grade_consistency(narrative, actual_grade)

        # [OPTIMASI #6] Safety post-filter
        narrative = safety_post_filter(narrative, inspection_data)

        return {
            "status": "success",
            "provider": "Google Gemini",
            "model": DEFAULT_GEMINI_MODEL,
            "role": "Auditor Mutu SNI 3926:2023",
            "text": narrative,
            "grade_consistency": "corrected" if had_conflict else "consistent"
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
        "text": offline_text,
        "grade_consistency": "consistent"
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
    Menggunakan SDK baru `google.genai`.
    [OPTIMASI #1, #3, #4, #5, #6, #7] Semua lapisan anti-halusinasi aktif.
    """
    # 1. Validasi awal batasan topik (Guardrail layer 1 + negative keywords)
    if not is_on_egg_topic(user_message) and len(user_message.split()) > 1:
        return get_out_of_context_warning_response()

    provider, api_key = detect_active_provider(custom_api_key)

    # Susun konteks inspeksi telur jika tersedia
    context_str = ""
    actual_grade = ""
    if inspection_context:
        summary = inspection_context.get("summary_360", {})
        grading = inspection_context.get("final_grading_decision", {})
        geo = inspection_context.get("opencv_geometry_inspection", {})
        actual_grade = grading.get("grade", "")
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
            f"{GROUNDING_INSTRUCTION}"  # [OPTIMASI #2] Grounding di chat juga
            f"--------------------------------------------------\n"
        )

    # 2. Mode Online dengan Gemini API (Aktif jika API key tersedia)
    if provider == "gemini" and genai is not None and genai_types is not None:
        model_candidates = [
            DEFAULT_GEMINI_MODEL,
            "gemini-3.5-flash",
            "gemini-3.5-flash-lite",
            "gemini-3.6-flash",
            "gemini-3-flash-preview",
            "gemini-3.8-flash",
            "gemini-3.7-flash"
        ]
        model_candidates = list(dict.fromkeys(model_candidates))
        last_error = ""

        # Bangun contents sebagai list of Content untuk multi-turn
        gemini_contents: List[genai_types.Content] = []

        if context_str:
            gemini_contents.append(genai_types.Content(
                role="user",
                parts=[genai_types.Part(text=f"Berikut data telur yang baru saja selesai diinspeksi:\n{context_str}\nPahami data di atas untuk menjawab pertanyaan-pertanyaan berikutnya. Ingat: HANYA gunakan data yang tersedia, jangan mengarang angka atau fakta baru.")]
            ))
            gemini_contents.append(genai_types.Content(
                role="model",
                parts=[genai_types.Part(text="Baik, data hasil inspeksi telur telah saya catat dan pahami sesuai acuan SNI 3926:2023. Saya akan menjawab pertanyaan HANYA berdasarkan data yang telah diberikan tanpa menambahkan informasi yang tidak ada. Silakan ajukan pertanyaan.")]
            ))

        # [OPTIMASI #4] Batasi history maksimal N turn terakhir
        if history:
            trimmed_history = history[-MAX_CHAT_HISTORY_TURNS * 2:]  # *2 karena setiap turn = user+model
            for turn in trimmed_history:
                role = "user" if turn.get("role") in ["user", "human"] else "model"
                content = turn.get("content", "").strip()
                if content:
                    gemini_contents.append(genai_types.Content(
                        role=role,
                        parts=[genai_types.Part(text=content)]
                    ))

        gemini_contents.append(genai_types.Content(
            role="user",
            parts=[genai_types.Part(text=user_message)]
        ))

        client = _create_gemini_client(api_key)
        if client is None:
            return {
                "status": "error",
                "is_warning": False,
                "reply": "Library 'google-genai' belum terpasang. Jalankan: pip install google-genai",
                "model": DEFAULT_GEMINI_MODEL,
                "provider": "Google Gemini"
            }

        config = _build_gen_config()

        for candidate in model_candidates:
            try:
                response = client.models.generate_content(
                    model=candidate,
                    contents=gemini_contents,
                    config=config,
                )
                if response and response.text:
                    reply_text = response.text.strip()

                    # [OPTIMASI #5] Validasi konsistensi grade pada chat juga
                    if actual_grade:
                        reply_text, _ = validate_grade_consistency(reply_text, actual_grade)

                    # [OPTIMASI #6] Safety post-filter
                    if inspection_context:
                        reply_text = safety_post_filter(reply_text, inspection_context)

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

        # Jika seluruh kandidat online gagal (misal koneksi terputus / kuota limit),
        # otomatis jatuh ke Mode 3 (Offline Fallback Cerdas) agar pengguna tidak terganggu error teknis
        print(f"[LLM WARNING] Seluruh model online Gemini gagal ({last_error}). Mengaktifkan Fallback Cerdas SNI.")

    # 3. Mode Offline Fallback Cerdas (SNI Knowledge Engine)
    # Menjawab pertanyaan standar SNI 3926:2023 secara cerdas dan akurat
    q_lower = user_message.lower()

    # Cek apakah user secara spesifik menanyakan TELUR AKTIF / TELUR INI yang sedang diinspeksi
    is_about_active_egg = any(w in q_lower for w in ["telur ini", "telur aktif", "telur sekarang", "telur tadi", "hasil inspeksi", "hasil scan", "telur yang baru"])

    if is_about_active_egg and inspection_context:
        grading = inspection_context.get("final_grading_decision", {})
        grade = grading.get("grade", "Mutu I (Grade A)")
        reason = grading.get("reason", "Cangkang utuh dan bersih")
        action = grading.get("hardware_sorting_action", "Kemas ke wadah standar")
        geo = inspection_context.get("opencv_geometry_inspection", {})
        summary = inspection_context.get("summary_360", {})

        reply = (
            f"Data hasil inspeksi telur aktif saat ini:\n\n"
            f"• **Vonis Grade**: **{grade}**\n"
            f"• **Cacat Kerabang (YOLOv8-seg)**: {summary.get('dominant_defect', 'good').upper()} (Catatan: {summary.get('notes', '-')})\n"
            f"• **Geometri & Bentuk (OpenCV)**: {geo.get('length_mm', 0)}×{geo.get('width_mm', 0)} mm, ESI: {geo.get('esi_percentage', 74)}% ({geo.get('shape_status', 'Normal')})\n"
            f"• **Estimasi Bobot**: ~{geo.get('estimated_weight_gram', 55)} gram (Kategori: {geo.get('size_category', 'Sedang')})\n"
            f"• **Alasan Regulasi**: {reason}\n"
            f"• **Aksi Konveyor/Servo**: {action}"
        )
        return {
            "status": "offline_fallback",
            "is_warning": False,
            "reply": reply,
            "model": "Rule-based Engine (Offline)",
            "provider": "Local Fallback"
        }

    # A. Pertanyaan Cacat Kerabang Retak / Pecah / Mutu III / Reject
    if any(k in q_lower for k in ["retak", "crack", "pecah", "bocor", "mutu iii", "mutu 3", "grade c", "reject", "afkir"]):
        reply = (
            "Berdasarkan standar **SNI 3926:2023 Tabel 1** (Syarat Mutu Kerabang Telur Konsumsi):\n\n"
            "• **Vonis Mutu**: Telur dengan cangkang retak (baik retak rambut / *hairline crack* maupun retak pecah terbuka) secara otomatis masuk kategori **Mutu III (Reject / Afkir)**.\n"
            "• **Bahaya & Keamanan Pangan**: Retakan merusak lapisan kutikula (*cuticle layer*) alami kerabang, sehingga bakteri patogen seperti *Salmonella spp.* dapat dengan cepat masuk mencemari putih dan kuning telur.\n"
            "• **Rekomendasi Penanganan**: Dilarang diperjualbelikan sebagai telur segar konsumsi meja. Sistem aktuator konveyor (pneumatik/servo) akan segera mengaktifkan pintu reject untuk memisahkan telur ke wadah karantina afkir."
        )
        return {
            "status": "offline_fallback",
            "is_warning": False,
            "reply": reply,
            "model": "Rule-based Engine (Offline)",
            "provider": "Local Fallback"
        }

    # B. Pertanyaan Noda Kotoran Cangkang / Mutu II
    if any(k in q_lower for k in ["kotor", "dirt", "noda", "feses", "bintik", "mutu ii", "mutu 2", "grade b"]):
        reply = (
            "Berdasarkan standar **SNI 3926:2023 Tabel 1** (Kebersihan Kerabang Telur):\n\n"
            "• **Toleransi Mutu II (Grade B)**: Kerabang utuh dan diperbolehkan memiliki sedikit noda kotoran ringan dengan luas maksimal **≤ 1/32 (≤ 3.125%)** dari total luas permukaan cangkang.\n"
            "• **Ambang Batas Reject (Mutu III)**: Jika luas noda kotoran feses/darah melebihi **> 1/16 (> 6.25%)**, telur wajib dialihkan ke Mutu III (Reject).\n"
            "• **Pemanfaatan & Distribusi**: Telur Mutu II dialihkan oleh aktuator Gate 2 sebagai bahan baku industri pangan olahan (seperti bakery, mie, kue) dengan perputaran stok cepat."
        )
        return {
            "status": "offline_fallback",
            "is_warning": False,
            "reply": reply,
            "model": "Rule-based Engine (Offline)",
            "provider": "Local Fallback"
        }

    # C. Pertanyaan Mutu I / Grade A (Prima & Bersih)
    if any(k in q_lower for k in ["mutu i", "mutu 1", "grade a", "super", "prima", "segar"]):
        reply = (
            "Berdasarkan standar **SNI 3926:2023** (Kriteria Mutu I / Grade A):\n\n"
            "1. **Kerabang**: Wajib utuh, halus, bebas dari retakan, dan bersih total dari noda kotoran atau feses (YOLOv8: *good*).\n"
            "2. **Indeks Bentuk (ESI)**: Berada dalam rentang proporsional ideal **72% – 76%**.\n"
            "3. **Bobot & Ukuran**: Memenuhi pengelompokan bobot SNI Tabel 2 (Kecil, Sedang, Besar, Ekstra Besar).\n\n"
            "Telur Mutu I adalah grade tertinggi dan dikemas langsung ke wadah ritel resmi berlabel untuk konsumsi segar."
        )
        return {
            "status": "offline_fallback",
            "is_warning": False,
            "reply": reply,
            "model": "Rule-based Engine (Offline)",
            "provider": "Local Fallback"
        }

    # D. Pertanyaan Bentuk / Egg Shape Index (ESI)
    if any(k in q_lower for k in ["shape", "bentuk", "esi", "indeks bentuk", "lonjong", "bulat"]):
        reply = (
            "Berdasarkan standar **SNI 3926:2023 Tabel 1** (Bentuk Telur Konsumsi):\n\n"
            "• **Bentuk Normal / Ideal**: Egg Shape Index (ESI) antara **72% – 76%** (rumus: Lebar ÷ Panjang × 100%).\n"
            "• **Bentuk Agak Lonjong**: ESI < 72% (telur lebih memanjang).\n"
            "• **Bentuk Agak Bulat**: ESI > 76% (telur cenderung membulat).\n\n"
            "Telur dengan ESI di luar rentang 72%–76% tidak dapat masuk Mutu I dan dialihkan ke **Mutu II (Grade B)** karena berisiko mudah pecah saat dikemas dalam baki (*egg tray*) standar."
        )
        return {
            "status": "offline_fallback",
            "is_warning": False,
            "reply": reply,
            "model": "Rule-based Engine (Offline)",
            "provider": "Local Fallback"
        }

    # E. Pertanyaan Bobot / Berat / Ukuran Telur
    if any(k in q_lower for k in ["berat", "bobot", "gram", "kategori ukuran", "tabel 2"]):
        reply = (
            "Berdasarkan **SNI 3926:2023 Tabel 2** (Klasifikasi Bobot Telur Ayam Konsumsi):\n\n"
            "• **Sangat Besar (Jumbo)**: Bobot > 60 gram per butir\n"
            "• **Besar**: Bobot 55 – 60 gram per butir\n"
            "• **Sedang**: Bobot 50 – 54.9 gram per butir\n"
            "• **Kecil**: Bobot < 50 gram per butir\n\n"
            "Sistem ini menggunakan estimasi geometri OpenCV (panjang × lebar) untuk mengestimasi bobot gram secara otomatis di atas konveyor."
        )
        return {
            "status": "offline_fallback",
            "is_warning": False,
            "reply": reply,
            "model": "Rule-based Engine (Offline)",
            "provider": "Local Fallback"
        }

    # F. Pertanyaan Pencucian / Sanitasi / Penyimpanan
    if any(k in q_lower for k in ["cuci", "pencucian", "simpan", "penyimpanan", "salmonella", "bakteri", "sanitasi"]):
        reply = (
            "Berdasarkan **SNI 3926:2023 Pasal 9 & 10** (Penanganan & Penyimpanan Telur):\n\n"
            "• **Pencucian**: Telur segar **tidak boleh dicuci dengan air biasa**, karena dapat melarutkan lapisan kutikula pelindung dan mempercepat masuknya bakteri *Salmonella sp.* ke dalam pori kerabang. Jika bernoda, gunakan metode pembersihan kering (*dry wiping*) steril.\n"
            "• **Penyimpanan**: Simpan telur pada suhu ruang ≤ 30°C dengan kelembaban relatif ≤ 90% (maksimal 14 hari), atau simpan pada suhu dingin 4°C – 10°C untuk masa simpan hingga 30 hari."
        )
        return {
            "status": "offline_fallback",
            "is_warning": False,
            "reply": reply,
            "model": "Rule-based Engine (Offline)",
            "provider": "Local Fallback"
        }

    # G. Pertanyaan Konveyor / Aktuator Sortir
    if any(k in q_lower for k in ["konveyor", "servo", "aktuator", "piston", "gate", "sortir"]):
        reply = (
            "Logika aktuator sortir mekanik pada sistem ini:\n\n"
            "• **Gate 1 (Mutu I / Grade A)**: Mendorong telur utuh bersih ke jalur pengemasan ritel berlabel.\n"
            "• **Gate 2 (Mutu II / Grade B)**: Mendorong telur bernoda ringan/bentuk non-ideal ke wadah industri olahan.\n"
            "• **Reject Ejector (Mutu III)**: Mendorong telur retak (*cracked*) atau noda parah ke wadah afkir karantina."
        )
        return {
            "status": "offline_fallback",
            "is_warning": False,
            "reply": reply,
            "model": "Rule-based Engine (Offline)",
            "provider": "Local Fallback"
        }

    # H. Pertanyaan Keamanan Pangan Khusus Telur Aktif
    if inspection_context and any(w in q_lower for w in ["aman", "konsumsi", "makan"]):
        grading = inspection_context.get("final_grading_decision", {})
        grade = grading.get("grade", "Mutu I (Grade A)")
        summary = inspection_context.get("summary_360", {})

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
        return {
            "status": "offline_fallback",
            "is_warning": False,
            "reply": reply,
            "model": "Rule-based Engine (Offline)",
            "provider": "Local Fallback"
        }

    # I. Jawaban default jika pertanyaan umum tentang telur belum terpetakan
    reply = (
        "Sebagai **Auditor Mutu AI SNI 3926:2023**, saya dapat membantu menjelaskan:\n"
        "1. Kriteria mutu kerabang: Mutu I (bebas noda), Mutu II (noda ≤ 1/32), Mutu III (retak/noda > 1/16).\n"
        "2. Indeks bentuk ESI normal (72% – 76%) dan rumus perhitungannya.\n"
        "3. Standar bobot telur ayam konsumsi (Tabel 2 SNI).\n"
        "4. Aspek keamanan pangan dan bahaya cemaran bakteri *Salmonella sp.* pada kerabang retak.\n\n"
        "Silakan ajukan pertanyaan spesifik terkait parameter di atas!"
    )

    return {
        "status": "offline_fallback",
        "is_warning": False,
        "reply": reply,
        "model": "Rule-based Engine (Offline)",
        "provider": "Local Fallback"
    }
