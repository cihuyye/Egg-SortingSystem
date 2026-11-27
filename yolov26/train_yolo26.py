# ==============================================================================
# Script Pelatihan Lokal Model YOLOv26-seg (Instance Segmentation)
# Sistem Sortir Mutu Telur SNI 3926:2023
# ==============================================================================

import os
import sys
import argparse
from pathlib import Path

_CURR_DIR = str(Path(__file__).resolve().parent)
_PARENT_DIR = str(Path(__file__).resolve().parent.parent)
if _CURR_DIR not in sys.path:
    sys.path.insert(0, _CURR_DIR)
if _PARENT_DIR not in sys.path:
    sys.path.append(_PARENT_DIR)

from ultralytics import YOLO


def resolve_data_yaml(path="Egg Dataset Clean/data.yaml"):
    if os.path.exists(path):
        return path
    alt = os.path.join(_PARENT_DIR, path)
    if os.path.exists(alt):
        return alt
    return path


def parse_args():
    parser = argparse.ArgumentParser(description="Pelatihan Model YOLOv26 Nano Segmentation")
    parser.add_argument("--model", type=str, default="yolo26n-seg.pt", help="Bobot dasar model")
    parser.add_argument("--data", type=str, default="Egg Dataset Clean/data.yaml", help="Path ke data.yaml")
    parser.add_argument("--epochs", type=int, default=80, help="Jumlah epoch pelatihan")
    parser.add_argument("--imgsz", type=int, default=640, help="Resolusi citra input")
    parser.add_argument("--batch", type=int, default=16, help="Ukuran batch")
    parser.add_argument("--device", type=str, default="0", help="Perangkat komputasi ('0' untuk GPU atau 'cpu')")
    parser.add_argument("--project", type=str, default="runs/segment", help="Direktori penyimpanan run")
    parser.add_argument("--name", type=str, default="egg_yolo26n_seg", help="Nama sub-folder eksperimen")
    return parser.parse_args()


def main():
    args = parse_args()
    data_path = resolve_data_yaml(args.data)

    print("=" * 75)
    print("🚀 PELATIHAN MODEL YOLOV26-SEG (NMS-FREE INSTANCE SEGMENTATION)")
    print(f"[*] Base Model : {args.model}")
    print(f"[*] Dataset    : {data_path}")
    print(f"[*] Epochs     : {args.epochs}")
    print(f"[*] Device     : {args.device}")
    print("=" * 75)

    model = YOLO(args.model)
    results = model.train(
        data=data_path,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        save=True,
        plots=True,
        exist_ok=True
    )

    print("\n" + "=" * 75)
    print("✅ Pelatihan selesai! Hasil tersimpan di:", os.path.join(args.project, args.name))
    print("=" * 75)


if __name__ == "__main__":
    main()
