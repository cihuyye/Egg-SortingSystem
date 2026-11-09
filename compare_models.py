"""
Script Komparasi Otomatis: YOLOv8 Nano vs RetinaNet (ResNet-50 FPN).
Melakukan evaluasi perbandingan head-to-head pada test set independen ('Egg Dataset Clean/test'):
1. Metrik Akurasi: mAP50, mAP50-95, Precision, Recall, F1-Score, Per-Class AP
2. Metrik Efisiensi: Model Size (MB), Parameter Count, Latensi Inferensi (ms), FPS
3. Visualisasi Komparasi: Side-by-Side [Ground Truth | YOLOv8 | RetinaNet]
4. Laporan Otomatis: Markdown report & Grafik perbandingan (Bar Chart & Scatter Plot)

Contoh Penggunaan:
    python compare_models.py --yolo-weights "Egg Sorting System/models/best.pt" --retina-weights "runs/retinanet/best_retinanet.pth"
"""

import os
import sys
import time
import argparse
import json
import csv
import cv2
import numpy as np
import torch
from PIL import Image
import torchvision.transforms.functional as TF
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from retinanet_dataset import EggYoloRetinaDataset, collate_fn
from retinanet_model import load_checkpoint
from retinanet_metrics import evaluate_detections


def parse_args():
    parser = argparse.ArgumentParser(description="Komparasi Head-to-Head YOLOv8 vs RetinaNet")
    parser.add_argument(
        "--yolo-weights",
        type=str,
        default="Egg Sorting System/models/best.pt",
        help="Path ke file bobot YOLOv8 (.pt)"
    )
    parser.add_argument(
        "--retina-weights",
        type=str,
        default="runs/retinanet/best_retinanet.pth",
        help="Path ke file bobot RetinaNet (.pth)"
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="Egg Dataset Clean",
        help="Path ke folder dataset"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["test", "valid"],
        help="Split dataset untuk pengujian (default: test)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="runs/comparison",
        help="Direktori penyimpanan hasil komparasi"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold untuk visualisasi (default: 0.25)"
    )
    parser.add_argument(
        "--eval-conf",
        type=float,
        default=0.05,
        help="Confidence threshold untuk perhitungan mAP (default: 0.05 standar COCO)"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Ukuran resolusi citra input (default: 640)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Perangkat inferensi ('cuda' atau 'cpu')"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Jumlah sampel citra untuk visualisasi side-by-side (default: 10)"
    )
    return parser.parse_args()


def count_parameters(model):
    """Menghitung total parameter model dalam jutaan (M)"""
    return sum(p.numel() for p in model.parameters()) / 1e6


def get_file_size_mb(file_path):
    """Mendapatkan ukuran file dalam MB"""
    if os.path.exists(file_path):
        return os.path.getsize(file_path) / (1024 * 1024)
    return 0.0


def draw_bounding_boxes(image_bgr, boxes, labels, scores, class_map, color_map):
    """Menggambar bounding box dan label pada citra OpenCV BGR"""
    img = image_bgr.copy()
    h, w = img.shape[:2]

    for b, l, s in zip(boxes, labels, scores):
        x1, y1, x2, y2 = [int(round(coord)) for coord in b]
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w - 1))
        y2 = max(0, min(y2, h - 1))

        cls_name = class_map.get(int(l), f"cls_{l}")
        color = color_map.get(cls_name, (0, 255, 0))

        # Kotak
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # Label tag
        tag = f"{cls_name} {s*100:.1f}%" if s is not None else cls_name
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(img, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, y1), color, -1)
        cv2.putText(img, tag, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    return img


def create_panel_with_header(img_bgr, title, header_color=(40, 40, 40)):
    """Menambahkan title header di atas citra"""
    h, w = img_bgr.shape[:2]
    header_h = 36
    panel = np.zeros((h + header_h, w, 3), dtype=np.uint8)
    panel[:header_h, :] = header_color
    panel[header_h:, :] = img_bgr

    # Teks header
    (tw, th), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    tx = max(10, (w - tw) // 2)
    ty = (header_h + th) // 2
    cv2.putText(panel, title, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    return panel


def generate_comparison_plots(yolo_metrics, retina_metrics, yolo_eff, retina_eff, output_dir):
    """Menghasilkan grafik komparasi akurasi dan efisiensi"""
    os.makedirs(output_dir, exist_ok=True)

    # 1. Bar Chart: Metrik Akurasi
    metrics_labels = ["mAP@0.50", "mAP@0.50:0.95", "Precision", "Recall", "F1-Score"]
    yolo_f1 = (2 * yolo_metrics["precision"] * yolo_metrics["recall"]) / max(1e-7, yolo_metrics["precision"] + yolo_metrics["recall"])
    retina_f1 = (2 * retina_metrics["precision"] * retina_metrics["recall"]) / max(1e-7, retina_metrics["precision"] + retina_metrics["recall"])

    yolo_vals = [
        yolo_metrics["mAP50"] * 100,
        yolo_metrics["mAP50_95"] * 100,
        yolo_metrics["precision"] * 100,
        yolo_metrics["recall"] * 100,
        yolo_f1 * 100
    ]
    retina_vals = [
        retina_metrics["mAP50"] * 100,
        retina_metrics["mAP50_95"] * 100,
        retina_metrics["precision"] * 100,
        retina_metrics["recall"] * 100,
        retina_f1 * 100
    ]

    x = np.arange(len(metrics_labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5.5))
    rects1 = ax.bar(x - width/2, yolo_vals, width, label="YOLOv8 Nano", color="#0284c7")
    rects2 = ax.bar(x + width/2, retina_vals, width, label="RetinaNet (ResNet-50 FPN)", color="#ea580c")

    ax.set_ylabel("Skor (%)", fontsize=11, fontweight="bold")
    ax.set_title("Komparasi Metrik Akurasi: YOLOv8 Nano vs RetinaNet", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_labels, fontsize=10, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_ylim(0, 115)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{height:.1f}%",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center", va="bottom", fontsize=9, fontweight="bold")

    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()
    bar_path = os.path.join(output_dir, "comparison_metrics_bar.png")
    plt.savefig(bar_path, dpi=300)
    plt.close()

    # 2. Scatter Plot: Speed vs Accuracy Trade-off
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(yolo_eff["latency_ms"], yolo_metrics["mAP50"] * 100, color="#0284c7", s=350, alpha=0.8, edgecolors="black", label=f"YOLOv8n ({yolo_eff['size_mb']:.1f} MB)")
    ax.scatter(retina_eff["latency_ms"], retina_metrics["mAP50"] * 100, color="#ea580c", s=700, alpha=0.8, edgecolors="black", label=f"RetinaNet ({retina_eff['size_mb']:.1f} MB)")

    ax.annotate(
        f"YOLOv8 Nano\n{yolo_eff['latency_ms']:.1f} ms | {yolo_eff['fps']:.0f} FPS\nmAP50: {yolo_metrics['mAP50']*100:.1f}%",
        xy=(yolo_eff["latency_ms"], yolo_metrics["mAP50"] * 100),
        xytext=(15, -15), textcoords="offset points",
        fontsize=9, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", fc="#e0f2fe", ec="#0284c7")
    )

    ax.annotate(
        f"RetinaNet (ResNet-50)\n{retina_eff['latency_ms']:.1f} ms | {retina_eff['fps']:.0f} FPS\nmAP50: {retina_metrics['mAP50']*100:.1f}%",
        xy=(retina_eff["latency_ms"], retina_metrics["mAP50"] * 100),
        xytext=(-30, -35), textcoords="offset points",
        fontsize=9, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", fc="#ffedd5", ec="#ea580c")
    )

    ax.set_xlabel("Latensi Inferensi (ms / citra) - Makin kiri makin cepat", fontsize=10, fontweight="bold")
    ax.set_ylabel("Akurasi mAP@0.50 (%) - Makin atas makin akurat", fontsize=10, fontweight="bold")
    ax.set_title("Analisis Trade-Off: Kecepatan vs Akurasi Deteksi Telur", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="lower right", fontsize=10)

    plt.tight_layout()
    scatter_path = os.path.join(output_dir, "speed_vs_accuracy.png")
    plt.savefig(scatter_path, dpi=300)
    plt.close()

    return bar_path, scatter_path


def main():
    args = parse_args()
    device = torch.device(args.device)

    print("=" * 80)
    print("⚖️  KOMPARASI HEAD-TO-HEAD: YOLOV8 NANO VS RETINANET (RESNET-50 FPN)")
    print("=" * 80)
    print(f"[*] YOLOv8 Weights   : {args.yolo_weights}")
    print(f"[*] RetinaNet Weights: {args.retina_weights}")
    print(f"[*] Dataset Dir      : {args.dataset_dir} (Split: {args.split.upper()})")
    print(f"[*] Perangkat        : {device}")
    print(f"[*] Output Dir       : {args.output_dir}")
    print("=" * 80)

    # 1. Validasi File Bobot
    if not os.path.exists(args.yolo_weights):
        raise FileNotFoundError(f"File bobot YOLOv8 tidak ditemukan: {args.yolo_weights}")
    if not os.path.exists(args.retina_weights):
        raise FileNotFoundError(
            f"File bobot RetinaNet tidak ditemukan di {args.retina_weights}!\n"
            f"Silakan latih terlebih dahulu dengan 'python train_retinanet.py' atau gunakan bobot dari Google Colab."
        )

    # 2. Load Dataset Test
    dataset = EggYoloRetinaDataset(
        root_dir=args.dataset_dir,
        split=args.split,
        imgsz=args.imgsz,
        augment=False
    )
    print(f"[*] Dataset pengujian dimuat: {len(dataset)} citra pada split '{args.split}'")

    # 3. Load YOLOv8 Model
    print("\n[1/4] Memuat model YOLOv8...")
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("Modul 'ultralytics' belum terinstall. Install dengan: pip install ultralytics")

    yolo_model = YOLO(args.yolo_weights)
    yolo_size_mb = get_file_size_mb(args.yolo_weights)
    yolo_params_m = count_parameters(yolo_model.model)

    # 4. Load RetinaNet Model
    print("[2/4] Memuat model RetinaNet...")
    retina_model, class_names = load_checkpoint(
        weights_path=args.retina_weights,
        device=device,
        score_thresh=args.eval_conf
    )
    retina_size_mb = get_file_size_mb(args.retina_weights)
    retina_params_m = count_parameters(retina_model)

    # 5. Evaluasi & Benchmarking Latensi
    print("\n[3/4] Menjalankan inferensi komparasi pada test set...")

    all_gt = []
    yolo_preds = []
    retina_preds = []

    yolo_latencies = []
    retina_latencies = []

    # Map kelas:
    # GT / RetinaNet: 1 -> cracked, 2 -> dirt
    # YOLOv8: 0 -> cracked, 1 -> dirt  (perlu +1 untuk evaluasi standar)
    color_map = {
        "cracked": (0, 0, 230),  # Merah
        "dirt": (0, 165, 255),    # Oranye
        "good": (0, 200, 0)      # Hijau
    }
    class_map = {1: "cracked", 2: "dirt"}

    side_by_side_dir = os.path.join(args.output_dir, "side_by_side")
    os.makedirs(side_by_side_dir, exist_ok=True)

    # Warm-up GPU
    if device.type == "cuda":
        dummy = torch.zeros((1, 3, args.imgsz, args.imgsz), device=device)
        for _ in range(5):
            _ = retina_model(dummy)

    for idx in range(len(dataset)):
        img_path = dataset.image_paths[idx]
        orig_img_bgr = cv2.imread(img_path)
        orig_h, orig_w = orig_img_bgr.shape[:2]

        # Ground truth
        lbl_path = dataset._find_label_path(img_path)
        gt_boxes, gt_labels = dataset._parse_label_file(lbl_path, orig_w, orig_h)
        all_gt.append({
            "boxes": torch.as_tensor(gt_boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.as_tensor(gt_labels, dtype=torch.int64)
        })

        # --- A. Inferensi YOLOv8 ---
        t_y0 = time.perf_counter()
        y_res = yolo_model.predict(source=img_path, conf=args.eval_conf, imgsz=args.imgsz, verbose=False, device=args.device)[0]
        t_y1 = time.perf_counter()
        yolo_latencies.append((t_y1 - t_y0) * 1000.0)

        y_boxes = y_res.boxes.xyxy.cpu() if len(y_res.boxes) > 0 else torch.zeros((0, 4))
        y_scores = y_res.boxes.conf.cpu() if len(y_res.boxes) > 0 else torch.zeros((0,))
        # Ubah class YOLO (0, 1) menjadi 1-indexed (1: cracked, 2: dirt) agar kompatibel
        y_labels = (y_res.boxes.cls.cpu().long() + 1) if len(y_res.boxes) > 0 else torch.zeros((0,), dtype=torch.int64)

        yolo_preds.append({
            "boxes": y_boxes,
            "scores": y_scores,
            "labels": y_labels
        })

        # --- B. Inferensi RetinaNet ---
        pil_im = Image.open(img_path).convert("RGB")
        resized_im = pil_im.resize((args.imgsz, args.imgsz), Image.BILINEAR)
        img_tensor = TF.to_tensor(resized_im).unsqueeze(0).to(device)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t_r0 = time.perf_counter()

        with torch.no_grad():
            r_res = retina_model(img_tensor)[0]

        if device.type == "cuda":
            torch.cuda.synchronize()
        t_r1 = time.perf_counter()
        retina_latencies.append((t_r1 - t_r0) * 1000.0)

        # Skala koordinat box kembali ke resolusi asli
        scale_x = orig_w / float(args.imgsz)
        scale_y = orig_h / float(args.imgsz)
        r_boxes = r_res["boxes"].cpu()
        if len(r_boxes) > 0:
            r_boxes[:, [0, 2]] *= scale_x
            r_boxes[:, [1, 3]] *= scale_y

        retina_preds.append({
            "boxes": r_boxes,
            "scores": r_res["scores"].cpu(),
            "labels": r_res["labels"].cpu()
        })

        # --- C. Buat Panel Visualisasi Side-by-Side untuk Sampel Terpilih ---
        if idx < args.num_samples or len(gt_boxes) > 0 and idx < (args.num_samples * 2):
            base_name = os.path.basename(img_path)

            # Panel 1: Ground Truth
            gt_drawn = draw_bounding_boxes(
                orig_img_bgr,
                gt_boxes,
                gt_labels,
                [None] * len(gt_boxes),
                class_map,
                color_map
            )
            panel_gt = create_panel_with_header(gt_drawn, "1. GROUND TRUTH (Label Manual)", header_color=(30, 41, 59))

            # Panel 2: YOLOv8
            y_vis_mask = y_scores >= args.conf
            y_drawn = draw_bounding_boxes(
                orig_img_bgr,
                y_boxes[y_vis_mask],
                y_labels[y_vis_mask],
                y_scores[y_vis_mask],
                class_map,
                color_map
            )
            panel_yolo = create_panel_with_header(y_drawn, "2. PREDIKSI YOLOV8 NANO", header_color=(3, 105, 161))

            # Panel 3: RetinaNet
            r_vis_mask = r_res["scores"].cpu() >= args.conf
            r_drawn = draw_bounding_boxes(
                orig_img_bgr,
                r_boxes[r_vis_mask],
                r_res["labels"].cpu()[r_vis_mask],
                r_res["scores"].cpu()[r_vis_mask],
                class_map,
                color_map
            )
            panel_retina = create_panel_with_header(r_drawn, "3. PREDIKSI RETINANET (RESNET-50)", header_color=(194, 65, 12))

            # Gabungkan 3 panel horizontal
            side_by_side = np.hstack([panel_gt, panel_yolo, panel_retina])
            out_sample_path = os.path.join(side_by_side_dir, f"compare_{idx+1:02d}_{base_name}")
            cv2.imwrite(out_sample_path, side_by_side)

    # 6. Hitung Metrik Deteksi
    print("[4/4] Menghitung metrik mAP, Precision, Recall...")
    yolo_metrics = evaluate_detections(yolo_preds, all_gt, class_ids=(1, 2))
    retina_metrics = evaluate_detections(retina_preds, all_gt, class_ids=(1, 2))

    yolo_lat_avg = float(np.mean(yolo_latencies))
    retina_lat_avg = float(np.mean(retina_latencies))

    yolo_eff = {
        "size_mb": yolo_size_mb,
        "params_m": yolo_params_m,
        "latency_ms": yolo_lat_avg,
        "fps": 1000.0 / yolo_lat_avg if yolo_lat_avg > 0 else 0.0
    }
    retina_eff = {
        "size_mb": retina_size_mb,
        "params_m": retina_params_m,
        "latency_ms": retina_lat_avg,
        "fps": 1000.0 / retina_lat_avg if retina_lat_avg > 0 else 0.0
    }

    yolo_f1 = (2 * yolo_metrics["precision"] * yolo_metrics["recall"]) / max(1e-7, yolo_metrics["precision"] + yolo_metrics["recall"])
    retina_f1 = (2 * retina_metrics["precision"] * retina_metrics["recall"]) / max(1e-7, retina_metrics["precision"] + retina_metrics["recall"])

    # 7. Tampilkan Ringkasan di Terminal
    print("\n" + "=" * 80)
    print("📊 TABEL PERBANDINGAN KOMPREHENSIF: YOLOV8 NANO VS RETINANET")
    print("=" * 80)
    print(f"{'Parameter Evaluasi':<32} | {'YOLOv8 Nano':<20} | {'RetinaNet (ResNet-50)':<22}")
    print("-" * 80)
    print(f"{'Backbone Network':<32} | {'Modified CSPDarknet':<20} | {'Deep ResNet-50 + FPN':<22}")
    print(f"{'Detection Head Paradigm':<32} | {'Anchor-Free Decoupled':<20} | {'Anchor-Based MultiScale':<22}")
    print(f"{'Loss Function':<32} | {'CIoU + DFL + BCE':<20} | {'Focal Loss + Smooth L1':<22}")
    print("-" * 80)
    print(f"{'mAP @ 0.50 (%)':<32} | {yolo_metrics['mAP50']*100:<20.2f} | {retina_metrics['mAP50']*100:<22.2f}")
    print(f"{'mAP @ 0.50:0.95 (%)':<32} | {yolo_metrics['mAP50_95']*100:<20.2f} | {retina_metrics['mAP50_95']*100:<22.2f}")
    print(f"{'Precision (%)':<32} | {yolo_metrics['precision']*100:<20.2f} | {retina_metrics['precision']*100:<22.2f}")
    print(f"{'Recall (%)':<32} | {yolo_metrics['recall']*100:<20.2f} | {retina_metrics['recall']*100:<22.2f}")
    print(f"{'F1-Score (%)':<32} | {yolo_f1*100:<20.2f} | {retina_f1*100:<22.2f}")
    print("-" * 80)
    print(f"{'AP50 - Cracked (Retak) (%)':<32} | {yolo_metrics['per_class_AP50']['cracked']*100:<20.2f} | {retina_metrics['per_class_AP50']['cracked']*100:<22.2f}")
    print(f"{'AP50 - Dirt (Kotoran) (%)':<32} | {yolo_metrics['per_class_AP50']['dirt']*100:<20.2f} | {retina_metrics['per_class_AP50']['dirt']*100:<22.2f}")
    print("-" * 80)
    print(f"{'Model File Size (MB)':<32} | {yolo_eff['size_mb']:<20.2f} | {retina_eff['size_mb']:<22.2f}")
    print(f"{'Parameter Count (Juta)':<32} | {yolo_eff['params_m']:<20.2f} | {retina_eff['params_m']:<22.2f}")
    print(f"{'Inference Latency (ms/citra)':<32} | {yolo_eff['latency_ms']:<20.2f} | {retina_eff['latency_ms']:<22.2f}")
    print(f"{'Inference Speed (FPS)':<32} | {yolo_eff['fps']:<20.1f} | {retina_eff['fps']:<22.1f}")
    print("=" * 80)

    # 8. Simpan Grafik Komparasi
    bar_img, scatter_img = generate_comparison_plots(
        yolo_metrics, retina_metrics, yolo_eff, retina_eff, args.output_dir
    )

    # 9. Generate Laporan Markdown Otomatis
    report_md_path = os.path.join(args.output_dir, "model_comparison_report.md")
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write(f"""# 🥚 Laporan Komparasi: YOLOv8 Nano vs RetinaNet (ResNet-50 FPN)
**Proyek:** Sistem Sortir Mutu Telur Otomatis Berdasarkan SNI 3926:2023  
**Dataset Pengujian:** `{args.dataset_dir}/{args.split}` ({len(dataset)} citra independen)  
**Perangkat:** `{device}` ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})  

---

## 1. Tabel Perbandingan Metrik Kinerja

| Parameter Komparasi | YOLOv8 Nano (`yolov8n.pt`) | RetinaNet (`ResNet-50 FPN`) | Selisih / Analisis |
| :--- | :---: | :---: | :---: |
| **Arsitektur Backbone** | Modified CSPDarknet53 | Deep ResNet-50 + FPN | ResNet-50 lebih dalam & berat |
| **Mekanisme Deteksi** | Anchor-Free Decoupled Head | Anchor-Based Multi-scale Head | YOLOv8 lebih modern & fleksibel |
| **Fungsi Loss** | CIoU + DFL + BCE | Focal Loss + Smooth L1 | Focal Loss unggul atasi imbalance |
| **mAP @ 0.50** | **{yolo_metrics['mAP50']*100:.2f}%** | **{retina_metrics['mAP50']*100:.2f}%** | {'+' if retina_metrics['mAP50'] >= yolo_metrics['mAP50'] else ''}{(retina_metrics['mAP50'] - yolo_metrics['mAP50'])*100:.2f}% |
| **mAP @ 0.50:0.95** | **{yolo_metrics['mAP50_95']*100:.2f}%** | **{retina_metrics['mAP50_95']*100:.2f}%** | {'+' if retina_metrics['mAP50_95'] >= yolo_metrics['mAP50_95'] else ''}{(retina_metrics['mAP50_95'] - yolo_metrics['mAP50_95'])*100:.2f}% |
| **Presisi (Precision)** | {yolo_metrics['precision']*100:.2f}% | {retina_metrics['precision']*100:.2f}% | Tingkat akurasi deteksi positif |
| **Recall** | {yolo_metrics['recall']*100:.2f}% | {retina_metrics['recall']*100:.2f}% | Sensitivitas menangkap cacat |
| **F1-Score** | {yolo_f1*100:.2f}% | {retina_f1*100:.2f}% | Harmonic mean presisi & recall |
| **AP50 Cracked (Retak)** | {yolo_metrics['per_class_AP50']['cracked']*100:.2f}% | {retina_metrics['per_class_AP50']['cracked']*100:.2f}% | Deteksi retakan rambut |
| **AP50 Dirt (Kotoran)** | {yolo_metrics['per_class_AP50']['dirt']*100:.2f}% | {retina_metrics['per_class_AP50']['dirt']*100:.2f}% | Deteksi noda kotoran |
| **Ukuran Bobot File** | **{yolo_eff['size_mb']:.2f} MB** | **{retina_eff['size_mb']:.2f} MB** | YOLOv8 ~{retina_eff['size_mb']/max(0.1, yolo_eff['size_mb']):.1f}x lebih ringkas |
| **Jumlah Parameter** | **{yolo_eff['params_m']:.2f} Juta** | **{retina_eff['params_m']:.2f} Juta** | RetinaNet memiliki ~{retina_eff['params_m']/max(0.1, yolo_eff['params_m']):.1f}x parameter |
| **Latensi Inferensi** | **{yolo_eff['latency_ms']:.2f} ms** | **{retina_eff['latency_ms']:.2f} ms** | YOLOv8 ~{retina_eff['latency_ms']/max(0.1, yolo_eff['latency_ms']):.1f}x lebih cepat |
| **Throughput (FPS)** | **{yolo_eff['fps']:.1f} FPS** | **{retina_eff['fps']:.1f} FPS** | Kemampuan proses real-time |

---

## 2. Analisis & Pembahasan Akademik (Untuk Laporan Proyek Semester 5)

### A. Karakteristik Arsitektur & Kompleksitas Komputasi
1. **YOLOv8 Nano (`yolov8n`)**:
   - Dirancang khusus untuk efisiensi komputasi tinggi pada edge device (Raspberry Pi, Jetson Nano, atau laptop biasa).
   - Menggunakan pendekatan *anchor-free*, sehingga tidak memerlukan penyesuaian ukuran anchor box buatan dan mampu menggeneralisasi bentuk cacat cangkang telur yang tidak beraturan dengan cepat.
   - Dengan parameter hanya ~{yolo_eff['params_m']:.1f}M dan ukuran bobot ~{yolo_eff['size_mb']:.1f} MB, inferensi dapat berjalan di atas konveyor dengan latensi sangat rendah (~{yolo_eff['latency_ms']:.1f} ms).

2. **RetinaNet (`ResNet-50 FPN`)**:
   - Model *one-stage detector* klasik yang menjadi tonggak penting dalam riset computer vision dengan memperkenalkan **Focal Loss** untuk mengatasi masalah ketidakseimbangan kelas (*foreground-background class imbalance*).
   - Memiliki representasi fitur visual yang sangat kaya berkat backbone ResNet-50 dan piramida fitur (FPN P3-P7).
   - Namun, kompleksitas parameternya mencapai ~{retina_eff['params_m']:.1f}M (~{retina_eff['size_mb']:.1f} MB), menyebabkan waktu inferensi lebih tinggi (~{retina_eff['latency_ms']:.1f} ms).

### B. Kesimpulan & Rekomendasi Deployment Sistem Sortir Telur
- **Untuk Implementasi Hardware Nyata (Real-time Conveyor):**  
  **YOLOv8 Nano direkomendasikan sebagai model utama** karena memenuhi syarat kecepatan inspeksi *real-time* konveyor (roller berputar 1–2 putaran per detik) tanpa menyebabkan jeda / bottleneck pada mekanik servo pemilah.
- **Untuk Verifikasi Mutu Resolusi Tinggi (Offline Quality Audit):**  
  **RetinaNet dapat digunakan sebagai model validasi sekunder** di server kontrol pusat untuk menganalisis retakan mikro atau sebagai benchmark pembanding dalam dokumen penelitian ilmiah / skripsi.

---
*Laporan ini dibuat otomatis oleh script `compare_models.py`*
""")

    print(f"\n[+] Hasil komparasi berhasil disimpan:")
    print(f"    - Laporan Markdown   : {report_md_path}")
    print(f"    - Grafik Bar Metrik  : {bar_img}")
    print(f"    - Grafik Trade-off   : {scatter_img}")
    print(f"    - Visual Side-by-Side: {side_by_side_dir}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()

