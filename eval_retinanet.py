"""
Script Evaluasi Independen Model RetinaNet pada Test Set Telur.
Menghasilkan metrik mAP50, mAP50-95, Precision, Recall, Per-class AP, serta kecepatan inferensi (FPS / ms).

Contoh Penggunaan:
    python eval_retinanet.py --weights runs/retinanet/best_retinanet.pth
"""

import os
import time
import json
import argparse
import torch
from torch.utils.data import DataLoader

from retinanet_dataset import EggYoloRetinaDataset, collate_fn
from retinanet_model import load_checkpoint
from retinanet_metrics import evaluate_detections


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluasi Independen Model RetinaNet")
    parser.add_argument(
        "--weights",
        type=str,
        default="runs/retinanet/best_retinanet.pth",
        help="Path ke file bobot checkpoint model (.pth)"
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
        choices=["test", "valid", "train"],
        help="Split dataset yang ingin dievaluasi (default: test)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size evaluasi"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Resolusi citra input"
    )
    parser.add_argument(
        "--score-thresh",
        type=float,
        default=0.05,
        help="Confidence score threshold untuk deteksi (default: 0.05 standar evaluasi mAP)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Perangkat inferensi ('cuda' atau 'cpu')"
    )
    return parser.parse_args()


def evaluate():
    args = parse_args()
    device = torch.device(args.device)

    print("=" * 70)
    print("🔍 EVALUASI INDEPENDEN MODEL RETINANET (RESNET-50 FPN)")
    print("=" * 70)
    print(f"[*] File Bobot   : {args.weights}")
    print(f"[*] Dataset Split: {args.split.upper()} di '{args.dataset_dir}'")
    print(f"[*] Perangkat    : {device}")
    print("=" * 70)

    # 1. Load Dataset
    test_dataset = EggYoloRetinaDataset(
        root_dir=args.dataset_dir,
        split=args.split,
        imgsz=args.imgsz,
        augment=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )

    print(f"[*] Total Citra Uji: {len(test_dataset)} citra ({len(test_loader)} batch)")

    # 2. Load Model
    print("[*] Memuat model dan bobot checkpoint...")
    model, class_names = load_checkpoint(
        weights_path=args.weights,
        device=device,
        score_thresh=args.score_thresh
    )

    # 3. Proses Inferensi & Benchmark Kecepatan
    all_preds = []
    all_gt = []
    latencies = []

    print("[*] Menjalankan inferensi pada seluruh sampel uji...")
    with torch.no_grad():
        for images, targets in test_loader:
            images = [img.to(device) for img in images]

            # Ukur latensi per batch
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            preds = model(images)

            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            batch_latency_ms = ((t1 - t0) / len(images)) * 1000.0
            latencies.extend([batch_latency_ms] * len(images))

            for p, t in zip(preds, targets):
                all_preds.append({
                    "boxes": p["boxes"].cpu(),
                    "scores": p["scores"].cpu(),
                    "labels": p["labels"].cpu()
                })
                all_gt.append({
                    "boxes": t["boxes"].cpu(),
                    "labels": t["labels"].cpu()
                })

    # 4. Hitung Metrik Deteksi
    metrics = evaluate_detections(all_preds, all_gt, class_ids=(1, 2))
    avg_latency = float(torch.tensor(latencies).mean().item())
    fps = 1000.0 / avg_latency if avg_latency > 0 else 0.0

    # Tampilkan Tabel Hasil
    print("\n" + "=" * 70)
    print("📊 HASIL EVALUASI MODEL RETINANET PADA DATASET UJI")
    print("=" * 70)
    print(f"{'Kelas':<15} | {'AP@0.50 (%)':<15} | {'AP@0.50:0.95 (%)':<18}")
    print("-" * 55)
    for cls_id, name in [(1, "cracked"), (2, "dirt")]:
        ap50 = metrics["per_class_AP50"][name] * 100
        ap50_95 = metrics["per_class_AP50_95"][name] * 100
        print(f"{name:<15} | {ap50:<15.2f} | {ap50_95:<18.2f}")
    print("-" * 55)
    print(f"{'ALL (mAP)':<15} | {metrics['mAP50'] * 100:<15.2f} | {metrics['mAP50_95'] * 100:<18.2f}")
    print("=" * 70)
    print(f"[*] Precision       : {metrics['precision'] * 100:.2f}%")
    print(f"[*] Recall          : {metrics['recall'] * 100:.2f}%")
    print(f"[*] Rata-rata Latensi: {avg_latency:.2f} ms / citra ({fps:.1f} FPS)")
    print("=" * 70)

    # Simpan Hasil ke JSON
    output_summary_path = os.path.splitext(args.weights)[0] + f"_eval_{args.split}.json"
    summary_data = {
        "weights": args.weights,
        "split": args.split,
        "total_images": len(test_dataset),
        "mAP50": round(metrics["mAP50"] * 100, 2),
        "mAP50_95": round(metrics["mAP50_95"] * 100, 2),
        "precision": round(metrics["precision"] * 100, 2),
        "recall": round(metrics["recall"] * 100, 2),
        "per_class_AP50": {k: round(v * 100, 2) for k, v in metrics["per_class_AP50"].items()},
        "per_class_AP50_95": {k: round(v * 100, 2) for k, v in metrics["per_class_AP50_95"].items()},
        "avg_latency_ms": round(avg_latency, 2),
        "fps": round(fps, 1)
    }

    with open(output_summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=4)

    print(f"[*] Ringkasan evaluasi disimpan di: {output_summary_path}\n")


if __name__ == "__main__":
    evaluate()

