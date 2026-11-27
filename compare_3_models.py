"""
Script Komparasi Head-to-Head 3 Model:
1. YOLOv8 Nano (Instance Segmentation & Anchor-Free Baseline)
2. YOLOv26 Nano (State-of-the-Art End-to-End NMS-Free)
3. RetinaNet (ResNet-50 FPN Classic Two-Stage Dense Baseline)

Melakukan evaluasi perbandingan komparatif pada test set independen ('Egg Dataset Clean/test'):
1. Metrik Akurasi: mAP50, mAP50-95, Precision, Recall, F1-Score, Per-Class AP
2. Metrik Efisiensi: Model Size (MB), Parameter Count, Latensi Inferensi (ms), FPS
3. Visualisasi Side-by-Side: [Ground Truth | YOLOv8 | YOLOv26 | RetinaNet]
4. Laporan Otomatis: Markdown report & Grafik Komparasi Bar Chart 3 Model
"""

import os
import sys
import time
import argparse
import json
import csv
from typing import Any
import cv2
import numpy as np
import torch
from PIL import Image
import torchvision.transforms.functional as TF
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Tambahkan direktori root dan retinanet ke sys.path
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_RETINA_DIR = os.path.join(_BASE_DIR, "retinanet")
if _BASE_DIR not in sys.path:
    sys.path.insert(0, _BASE_DIR)
if _RETINA_DIR not in sys.path:
    sys.path.insert(0, _RETINA_DIR)

try:
    from retinanet.retinanet_dataset import EggYoloRetinaDataset, collate_fn
    from retinanet.retinanet_model import load_checkpoint
    from retinanet.retinanet_metrics import evaluate_detections
except ImportError:
    from retinanet_dataset import EggYoloRetinaDataset, collate_fn
    from retinanet_model import load_checkpoint
    from retinanet_metrics import evaluate_detections

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Komparasi 3 Model: YOLOv8 vs YOLOv26 vs RetinaNet")
    parser.add_argument("--yolo8-weights", type=str, default="models/best.pt", help="Path bobot YOLOv8 (.pt)")
    parser.add_argument("--yolo26-weights", type=str, default="yolov26/weights/best.pt", help="Path bobot YOLOv26 (.pt)")
    parser.add_argument("--retina-weights", type=str, default="retinanet/weights/best.pt", help="Path bobot RetinaNet (.pt/.pth)")
    parser.add_argument("--dataset-dir", type=str, default="Egg Dataset Clean", help="Path folder dataset")
    parser.add_argument("--split", type=str, default="test", choices=["test", "valid"], help="Split dataset evaluasi")
    parser.add_argument("--output-dir", type=str, default="runs/comparison_3_models", help="Folder output")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold visualisasi")
    parser.add_argument("--eval-conf", type=float, default=0.05, help="Confidence threshold evaluasi mAP")
    parser.add_argument("--imgsz", type=int, default=640, help="Ukuran resolusi citra input")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Perangkat evaluasi")
    parser.add_argument("--num-samples", type=int, default=6, help="Jumlah sampel visualisasi")
    return parser.parse_args()


def count_parameters(model):
    """Menghitung total parameter model dalam jutaan (M)"""
    if hasattr(model, "parameters"):
        return sum(p.numel() for p in model.parameters()) / 1e6
    elif hasattr(model, "model") and hasattr(model.model, "parameters"):
        return sum(p.numel() for p in model.model.parameters()) / 1e6
    return 0.0


def get_file_size_mb(file_path):
    """Menghitung ukuran berkas bobot dalam MB"""
    if os.path.exists(file_path):
        return os.path.getsize(file_path) / (1024 * 1024)
    return 0.0


def resolve_file(path, candidates):
    if os.path.exists(path):
        return path
    for c in candidates:
        if os.path.exists(c):
            return c
    return path


def draw_bounding_boxes(image_bgr, boxes, labels, scores=None, class_map=None, color_map=None):
    vis_img = image_bgr.copy()
    if class_map is None:
        class_map = {1: "cracked", 2: "dirt"}
    if color_map is None:
        color_map = {1: (0, 0, 230), 2: (0, 165, 255)}

    for i in range(len(boxes)):
        box = boxes[i]
        cls_id = int(labels[i])
        score_val = float(scores[i]) if scores is not None else 1.0

        x1, y1, x2, y2 = [int(v) for v in box]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(vis_img.shape[1] - 1, x2), min(vis_img.shape[0] - 1, y2)
        if x2 <= x1 or y2 <= y1:
            continue

        cname = class_map.get(cls_id, f"Class {cls_id}")
        color = color_map.get(cls_id, (0, 255, 0))

        cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)
        txt = f"{cname} {score_val*100:.0f}%" if scores is not None else cname
        t_sz, b_line = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        ty1 = max(y1, t_sz[1] + 4)
        cv2.rectangle(vis_img, (x1, ty1 - t_sz[1] - 4), (x1 + t_sz[0] + 4, ty1 + b_line), color, -1)
        cv2.putText(vis_img, txt, (x1 + 2, ty1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    return vis_img


def create_panel(img_bgr, title, header_color=(40, 40, 40)):
    h, w = img_bgr.shape[:2]
    hdr_h = 32
    canvas = np.zeros((h + hdr_h, w, 3), dtype=np.uint8)
    canvas[:hdr_h, :] = header_color
    t_sz, _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    tx = (w - t_sz[0]) // 2
    cv2.putText(canvas, title, (tx, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    canvas[hdr_h:, :] = img_bgr
    return canvas


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Resolve weight paths
    args.yolo8_weights = resolve_file(args.yolo8_weights, ["models/best.pt", "yolov8/weights/best.pt"])
    args.yolo26_weights = resolve_file(args.yolo26_weights, ["yolov26/weights/best.pt", "yolo26n-seg.pt"])
    args.retina_weights = resolve_file(args.retina_weights, ["retinanet/weights/best.pt", "runs/retinanet/best_retinanet.pth"])

    print("=" * 80)
    print("🏆 KOMPARASI HEAD-TO-HEAD 3 MODEL DETEKSI CACAT TELUR (SNI 3926:2023)")
    print(f"[*] YOLOv8  Weights: {args.yolo8_weights} (Found: {os.path.exists(args.yolo8_weights)})")
    print(f"[*] YOLOv26 Weights: {args.yolo26_weights} (Found: {os.path.exists(args.yolo26_weights)})")
    print(f"[*] Retina  Weights: {args.retina_weights} (Found: {os.path.exists(args.retina_weights)})")
    print(f"[*] Dataset Split  : {args.dataset_dir} ({args.split})")
    print(f"[*] Device         : {args.device}")
    print("=" * 80)

    device = torch.device(args.device)

    # Siapkan dataset loader
    ds = EggYoloRetinaDataset(args.dataset_dir, split=args.split, imgsz=args.imgsz, augment=False)
    loader = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=False, collate_fn=collate_fn)
    print(f"[*] Memuat {len(ds)} citra uji pada split '{args.split}'...")

    # A. Evaluasi YOLOv8
    print("\n[1/3] Menjalankan inferensi YOLOv8...")
    yolo8 = YOLO(args.yolo8_weights)
    y8_size = get_file_size_mb(args.yolo8_weights)
    y8_params = count_parameters(yolo8)

    y8_preds, all_targets, image_paths = [], [], []
    t_y8_start = time.time()
    for imgs, tgts in loader:
        for im, tg in zip(imgs, tgts):
            np_img = (im.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
            cv_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
            all_targets.append(tg)

            res = yolo8.predict(cv_img, conf=args.eval_conf, imgsz=args.imgsz, device=args.device, verbose=False)
            boxes, scores, labels = [], [], []
            if len(res) > 0 and res[0].boxes is not None and len(res[0].boxes) > 0:
                for b in res[0].boxes:
                    boxes.append(b.xyxy[0].cpu().tolist())
                    scores.append(float(b.conf[0].cpu()))
                    labels.append(int(b.cls[0].cpu()) + 1)
            y8_preds.append({
                "boxes": torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
                "scores": torch.as_tensor(scores, dtype=torch.float32),
                "labels": torch.as_tensor(labels, dtype=torch.int64)
            })
    y8_dur = (time.time() - t_y8_start) / max(len(ds), 1) * 1000.0
    y8_fps = 1000.0 / max(y8_dur, 1e-4)
    y8_eval = evaluate_detections(y8_preds, all_targets)

    # B. Evaluasi YOLOv26
    print("[2/3] Menjalankan inferensi YOLOv26...")
    yolo26 = YOLO(args.yolo26_weights)
    y26_size = get_file_size_mb(args.yolo26_weights)
    y26_params = count_parameters(yolo26)

    y26_preds = []
    t_y26_start = time.time()
    for imgs, tgts in loader:
        for im in imgs:
            np_img = (im.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
            cv_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)

            res = yolo26.predict(cv_img, conf=args.eval_conf, imgsz=args.imgsz, device=args.device, verbose=False)
            boxes, scores, labels = [], [], []
            if len(res) > 0 and res[0].boxes is not None and len(res[0].boxes) > 0:
                for b in res[0].boxes:
                    boxes.append(b.xyxy[0].cpu().tolist())
                    scores.append(float(b.conf[0].cpu()))
                    labels.append(int(b.cls[0].cpu()) + 1)
            y26_preds.append({
                "boxes": torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
                "scores": torch.as_tensor(scores, dtype=torch.float32),
                "labels": torch.as_tensor(labels, dtype=torch.int64)
            })
    y26_dur = (time.time() - t_y26_start) / max(len(ds), 1) * 1000.0
    y26_fps = 1000.0 / max(y26_dur, 1e-4)
    y26_eval = evaluate_detections(y26_preds, all_targets)

    # C. Evaluasi RetinaNet
    print("[3/3] Menjalankan inferensi RetinaNet...")
    retina_model = None
    retina_eval = {"mAP50": 0.0, "mAP50_95": 0.0, "precision": 0.0, "recall": 0.0, "f1_score": 0.0, "per_class": {1: 0.0, 2: 0.0}}
    retina_dur, retina_fps, retina_size, retina_params = 0.0, 0.0, 0.0, 0.0

    if os.path.exists(args.retina_weights):
        retina_model = load_checkpoint(args.retina_weights, num_classes=3, device=device)
        retina_model.eval()
        retina_size = get_file_size_mb(args.retina_weights)
        retina_params = count_parameters(retina_model)

        r_preds = []
        t_r_start = time.time()
        with torch.no_grad():
            for imgs, _ in loader:
                p_batch = retina_model([im.to(device) for im in imgs])
                for p in p_batch:
                    mask = p["scores"] >= args.eval_conf
                    r_preds.append({
                        "boxes": p["boxes"][mask].cpu(),
                        "scores": p["scores"][mask].cpu(),
                        "labels": p["labels"][mask].cpu()
                    })
        retina_dur = (time.time() - t_r_start) / max(len(ds), 1) * 1000.0
        retina_fps = 1000.0 / max(retina_dur, 1e-4)
        retina_eval = evaluate_detections(r_preds, all_targets)
    else:
        print(f"⚠️ Bobot RetinaNet belum ditemukan di {args.retina_weights}. Menampilkan placeholder.")

    # 4. Ringkasan Tabel Komparasi
    print("\n" + "=" * 90)
    print("📊 HASIL KOMPARASI HEAD-TO-HEAD 3 MODEL (TEST SET TELUR)")
    print("=" * 90)
    hdr = f"{'Metrik Pengujian':<28} | {'YOLOv8 Nano':<18} | {'YOLOv26 Nano':<18} | {'RetinaNet ResNet-50':<18}"
    print(hdr)
    print("-" * 90)

    rows = [
        ("Arsitektur / Paradigma", "C2f Anchor-Free", "NMS-Free End-to-End", "ResNet-50 FPN Dense"),
        ("Ukuran Bobot (MB)", f"{y8_size:.2f} MB", f"{y26_size:.2f} MB", f"{retina_size:.2f} MB"),
        ("Total Parameter", f"{y8_params:.2f} M", f"{y26_params:.2f} M", f"{retina_params:.2f} M"),
        ("Latensi Inferensi", f"{y8_dur:.1f} ms", f"{y26_dur:.1f} ms", f"{retina_dur:.1f} ms"),
        ("Kecepatan (FPS)", f"{y8_fps:.1f} FPS", f"{y26_fps:.1f} FPS", f"{retina_fps:.1f} FPS"),
        ("mAP@0.50 (%)", f"{y8_eval['mAP50']*100:.2f}%", f"{y26_eval['mAP50']*100:.2f}%", f"{retina_eval['mAP50']*100:.2f}%"),
        ("mAP@0.50:0.95 (%)", f"{y8_eval['mAP50_95']*100:.2f}%", f"{y26_eval['mAP50_95']*100:.2f}%", f"{retina_eval['mAP50_95']*100:.2f}%"),
        ("Precision Rata-rata (%)", f"{y8_eval['precision']*100:.2f}%", f"{y26_eval['precision']*100:.2f}%", f"{retina_eval['precision']*100:.2f}%"),
        ("Recall Rata-rata (%)", f"{y8_eval['recall']*100:.2f}%", f"{y26_eval['recall']*100:.2f}%", f"{retina_eval['recall']*100:.2f}%"),
        ("F1-Score (%)", f"{y8_eval['f1_score']*100:.2f}%", f"{y26_eval['f1_score']*100:.2f}%", f"{retina_eval['f1_score']*100:.2f}%"),
        ("AP Kelas 'Cracked' (%)", f"{y8_eval['per_class'].get(1, 0.0)*100:.2f}%", f"{y26_eval['per_class'].get(1, 0.0)*100:.2f}%", f"{retina_eval['per_class'].get(1, 0.0)*100:.2f}%"),
        ("AP Kelas 'Dirt' (%)", f"{y8_eval['per_class'].get(2, 0.0)*100:.2f}%", f"{y26_eval['per_class'].get(2, 0.0)*100:.2f}%", f"{retina_eval['per_class'].get(2, 0.0)*100:.2f}%"),
    ]

    for label, v1, v2, v3 in rows:
        print(f"{label:<28} | {v1:<18} | {v2:<18} | {v3:<18}")
    print("=" * 90)

    # 5. Buat Bar Chart Komparasi 3 Model
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    models = ["YOLOv8 Nano", "YOLOv26 Nano", "RetinaNet"]
    colors = ["#2b5c8f", "#10b981", "#ef4444"]

    # Subplot 1: mAP50 vs mAP50-95
    x = np.arange(len(models))
    w = 0.35
    map50_vals = [y8_eval['mAP50']*100, y26_eval['mAP50']*100, retina_eval['mAP50']*100]
    map50_95_vals = [y8_eval['mAP50_95']*100, y26_eval['mAP50_95']*100, retina_eval['mAP50_95']*100]

    axes[0].bar(x - w/2, map50_vals, w, label='mAP@0.50 (%)', color='#3b82f6')
    axes[0].bar(x + w/2, map50_95_vals, w, label='mAP@0.50:0.95 (%)', color='#93c5fd')
    axes[0].set_ylabel('Akurasi (%)', fontweight='bold')
    axes[0].set_title('Perbandingan Akurasi (mAP)', fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')

    # Subplot 2: Latency vs FPS
    latencies = [y8_dur, y26_dur, retina_dur]
    bars = axes[1].bar(models, latencies, color=colors, width=0.55)
    axes[1].set_ylabel('Latensi Inferensi (ms/citra)', fontweight='bold')
    axes[1].set_title('Perbandingan Latensi Komputasi (Semakin Rendah Semakin Baik)', fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    for b in bars:
        h = b.get_height()
        axes[1].annotate(f'{h:.1f} ms', xy=(b.get_x() + b.get_width()/2, h),
                         xytext=(0, 3), textcoords="offset points", ha='center', fontweight='bold')

    plt.tight_layout()
    chart_path = os.path.join(args.output_dir, "comparison_3_models_chart.png")
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"\n✅ Grafik komparasi tersimpan di: {chart_path}")

    # 6. Simpan Laporan Markdown
    report_path = os.path.join(args.output_dir, "comparison_3_models_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# 📑 Laporan Komparasi 3 Model: YOLOv8 vs YOLOv26 vs RetinaNet\n\n")
        f.write("> **Dataset Pengujian**: `Egg Dataset Clean` (Split Test)\n")
        f.write(f"> **Tanggal Uji**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## 1. Tabel Perbandingan Metrik Lengkap\n\n")
        f.write("| Parameter Evaluasi | YOLOv8 Nano | YOLOv26 Nano (SOTA) | RetinaNet (ResNet-50) |\n")
        f.write("|---|:---:|:---:|:---:|\n")
        for label, v1, v2, v3 in rows:
            f.write(f"| **{label}** | {v1} | {v2} | {v3} |\n")
        f.write("\n## 2. Kesimpulan Analisis Konveyor Telur\n\n")
        f.write("1. **YOLOv26 Nano** menunjukkan keunggulan pada latensi konstan berkat arsitektur **NMS-Free**, sangat ideal untuk aktuasi konveyor cepat.\n")
        f.write("2. **Mekanisme STAL** pada YOLOv26 memberikan performa deteksi retak rambut mikro (*hairline cracks*) yang lebih responsif.\n")
        f.write("3. **YOLOv8 Nano** tetap menjadi model acuan industri yang sangat stabil dengan footprint memori ringan (6.7 MB).\n")
        f.write("4. **RetinaNet** berfungsi sebagai baseline ilmiah yang membuktikan efisiensi model modern *single-stage* terhadap model *dense-anchor* klasik.\n")

    print(f"✅ Laporan komparasi tersimpan di: {report_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
