# ==============================================================================
# Script Inferensi & Visualisasi Segmentasi Model YOLOv26-seg
# Menghitung Rasio Luas Noda Cangkang untuk Penentuan Mutu SNI 3926:2023
# ==============================================================================

import os
import sys
import argparse
from pathlib import Path
import cv2
import numpy as np

_CURR_DIR = str(Path(__file__).resolve().parent)
_PARENT_DIR = str(Path(__file__).resolve().parent.parent)
if _CURR_DIR not in sys.path:
    sys.path.insert(0, _CURR_DIR)
if _PARENT_DIR not in sys.path:
    sys.path.append(_PARENT_DIR)

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Inferensi Model YOLOv26-seg")
    parser.add_argument(
        "--source",
        type=str,
        default="Egg Dataset Clean/test/images/cracked",
        help="Path ke citra atau folder citra"
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="yolov26/weights/best.pt",
        help="Path ke bobot model YOLOv26-seg"
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Resolusi inferensi")
    parser.add_argument("--device", type=str, default="cpu", help="'cuda' atau 'cpu'")
    parser.add_argument("--output", type=str, default="runs/predict_yolo26", help="Folder output")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    weights_path = args.weights
    if not os.path.exists(weights_path):
        alt = os.path.join(_PARENT_DIR, weights_path)
        if os.path.exists(alt):
            weights_path = alt
        else:
            print(f"⚠️ Bobot {weights_path} belum ditemukan. Menggunakan pre-trained 'yolo26n-seg.pt'...")
            weights_path = "yolo26n-seg.pt"

    print("=" * 70)
    print("🔍 INFERENSI YOLOV26-SEGMENTATION PADA SAMPEL TELUR")
    print(f"[*] Weights : {weights_path}")
    print(f"[*] Source  : {args.source}")
    print("=" * 70)

    model = YOLO(weights_path)
    results = model.predict(
        source=args.source,
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
        save=True,
        project=args.output,
        name="visualisasi",
        exist_ok=True
    )

    print(f"✅ Prediksi selesai! Gambar visualisasi disimpan di: {os.path.join(args.output, 'visualisasi')}")


if __name__ == "__main__":
    main()
