import time
from typing import Any, Dict, List
import cv2
import numpy as np
from modules.candling_detector import process_candling_frame

def run_candling_dashboard(camera_index: int = 0) -> None:
    """
    Dashboard Uji Nyata Candling Telur (Webcam / Side-View / Top-View).
    Fitur Unggulan:
    1. Otomatis mengenali orientasi telur (Tidur Horizontal Kiri/Kanan, atau Berdiri).
    2. Mendeteksi kutub tumpul (blunt end) tempat kantung hawa berada secara adaptif.
    3. Filter bintik pigmen alami & noise kamera gelap (anti false-positive).
    4. Mengeluarkan Vektor Fitur & Status SNI 3926:2023 stabil tiap detik.
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"[ERROR] Kamera index {camera_index} tidak dapat dibuka.")
        return

    print("=" * 65)
    print("  DASHBOARD LIVE CANDLING TELUR BERDASARKAN SNI 3926:2023")
    print("  Tekan 'q' pada jendela video untuk keluar")
    print("=" * 65)

    last_eval_time: float = time.time()
    current_vector: List[float] = [0.0, 0.0, 0.0]
    current_grade: str = "Menunggu..."
    current_orientation: str = "-"
    current_reason: str = ""

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        now: float = time.time()

        # Pemrosesan langsung frame BGR OpenCV dengan fungsi yang dioptimalkan
        res: Dict[str, Any]
        visual_frame: np.ndarray
        res, visual_frame = process_candling_frame(frame, px_to_mm=0.15)

        # Update data stabil tiap 1 detik ke terminal
        if (now - last_eval_time) >= 1.0:
            last_eval_time = now
            air_depth: float = float(res.get("air_cell_depth_cm", 0.0))
            dark_ratio: float = float(res.get("dark_spot_ratio", 0.0))
            brightness: float = float(res.get("mean_brightness", 0.0))
            current_grade = str(res.get("candling_status", "Tidak Diketahui"))
            current_orientation = str(res.get("orientation_detected", "-"))
            current_reason = str(res.get("reason", ""))

            # Vektor Candling: [Kedalaman Kantung (cm), Rasio Cacat Gelap, Kecerahan]
            current_vector = [round(air_depth, 2), round(dark_ratio, 4), round(brightness, 1)]
            
            timestamp: str = time.strftime("%H:%M:%S")
            print(f"[{timestamp}] Orientasi: {current_orientation}")
            print(f"           Vektor Candling -> {current_vector}")
            print(f"           Grade SNI: {current_grade} | {current_reason}\n")

        # Overlay Informasi Dashboard di Layar
        h_f, w_f = visual_frame.shape[:2]
        overlay: np.ndarray = visual_frame.copy()
        cv2.rectangle(overlay, (0, 0), (int(w_f), 95), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.65, visual_frame, 0.35, 0.0, visual_frame)

        # Teks Info
        color_grade = (0, 255, 0) if "Mutu I" in current_grade else ((0, 255, 255) if "Mutu II" in current_grade else (0, 0, 255))
        cv2.putText(visual_frame, f"Vektor Candling: {current_vector} (Depth_cm, Dark_ratio, Brightness)", (15, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.putText(visual_frame, f"Orientasi : {current_orientation}", (15, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(visual_frame, f"Status SNI: {current_grade}", (15, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color_grade, 2)

        cv2.imshow("Dashboard Deteksi Interior Telur (Candling)", visual_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_candling_dashboard(0)
