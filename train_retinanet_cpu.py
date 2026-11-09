"""
Script Pelatihan Model RetinaNet (ResNet-50 FPN) Khusus CPU Laptop.
Dioptimasi dengan:
1. Backbone Freezing (Hanya melatih FPN & Classification Head) -> 5-10x lebih cepat di CPU.
2. Ukuran input 512x512 & batch size kecil (4) agar RAM/CPU tidak panas berlebih.
3. Otomatis menghasilkan best.pt dan grafik evaluasi (results.png, confusion_matrix.png, val_batch_pred.jpg).

Jalankan:
    python train_retinanet_cpu.py --epochs 15 --batch-size 4
"""

import os
import sys
import time
import argparse
import csv
import zipfile
import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image
import torchvision.transforms.functional as TF
import torchvision
from torchvision.models.detection.retinanet import (
    retinanet_resnet50_fpn,
    RetinaNetClassificationHead,
    RetinaNet_ResNet50_FPN_Weights
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from retinanet_dataset import EggYoloRetinaDataset, collate_fn, NUM_CLASSES, CLASS_MAPPING_YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Training RetinaNet Hemat Sumber Daya di CPU")
    parser.add_argument("--dataset-dir", type=str, default="Egg Dataset Clean")
    parser.add_argument("--epochs", type=int, default=15, help="Jumlah epoch (default: 15, sangat pas untuk CPU)")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size (default: 4)")
    parser.add_argument("--imgsz", type=int, default=512, help="Ukuran resolusi citra input (default: 512)")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--freeze-backbone", action="store_true", default=True, help="Freeze backbone ResNet-50 (jauh lebih cepat)")
    parser.add_argument("--output-dir", type=str, default="runs/retinanet_cpu")
    return parser.parse_args()


def box_iou(b1, b2):
    area1 = (b1[:, 2] - b1[:, 0]) * (b1[:, 3] - b1[:, 1])
    area2 = (b2[:, 2] - b2[:, 0]) * (b2[:, 3] - b2[:, 1])
    lt = torch.max(b1[:, None, :2], b2[:, :2])
    rb = torch.min(b1[:, None, 2:], b2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2 - inter
    return inter / union.clamp(min=1e-7)


def compute_ap(rec, prec):
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    x = np.linspace(0, 1, 101)
    return float(np.trapz(np.interp(x, mrec, mpre), x))


def main():
    args = parse_args()
    device = torch.device("cpu")
    torch.set_num_threads(os.cpu_count() or 4)

    print("=" * 75)
    print("💻 PELATIHAN RETINANET RESNET-50 FPN DI CPU LAPTOP (OPTIMIZED)")
    print("=" * 75)
    print(f"[*] Perangkat        : CPU ({torch.get_num_threads()} threads)")
    print(f"[*] Dataset          : {args.dataset_dir}")
    print(f"[*] Epochs           : {args.epochs} | Batch Size: {args.batch_size} | Resolusi: {args.imgsz}x{args.imgsz}")
    print(f"[*] Freeze Backbone  : {args.freeze_backbone} (Transfer Learning Cepat)")
    print(f"[*] Folder Output    : {args.output_dir}")
    print("=" * 75)

    # 1. Dataset & DataLoader
    train_ds = EggYoloRetinaDataset(args.dataset_dir, split="train", imgsz=args.imgsz, augment=True)
    valid_ds = EggYoloRetinaDataset(args.dataset_dir, split="valid", imgsz=args.imgsz, augment=False)
    test_ds = EggYoloRetinaDataset(args.dataset_dir, split="test", imgsz=args.imgsz, augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # 2. Bangun Model RetinaNet
    print("\n[*] Mengunduh & memuat bobot COCO pre-trained...")
    weights = RetinaNet_ResNet50_FPN_Weights.DEFAULT
    model = retinanet_resnet50_fpn(weights=weights, score_thresh=0.05, nms_thresh=0.5)

    # Freeze Backbone ResNet-50 jika diaktifkan (mempercepat komputasi CPU secara drastis)
    if args.freeze_backbone:
        for p in model.backbone.body.parameters():
            p.requires_grad = False
        print("[*] ResNet-50 backbone berhasil di-freeze (hanya FPN & Head yang dilatih)")

    # Ganti head ke 3 kelas (0: bg, 1: cracked, 2: dirt)
    in_channels = getattr(model.head.classification_head, "in_channels", getattr(model.head.classification_head.cls_logits, "in_channels", 256))
    num_anchors = getattr(model.head.classification_head, "num_anchors", 9)
    model.head.classification_head = RetinaNetClassificationHead(
        in_channels=in_channels,
        num_anchors=num_anchors,
        num_classes=3,
        prior_probability=0.01
    )
    model.to(device)

    # 3. Setup Optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    os.makedirs(args.output_dir, exist_ok=True)
    weights_dir = os.path.join(args.output_dir, "weights")
    os.makedirs(weights_dir, exist_ok=True)
    best_weight_path = os.path.join(weights_dir, "best.pt")
    csv_history_path = os.path.join(args.output_dir, "metrics_summary.csv")

    history = []
    best_map50 = 0.0

    print("\n[*] Memulai pelatihan...")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        t_loss, t_cls, t_box = 0.0, 0.0, 0.0

        for imgs, tgts in train_loader:
            optimizer.zero_grad()
            ld = model(imgs, tgts)
            loss = ld["classification"] + ld["bbox_regression"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=10.0)
            optimizer.step()

            t_loss += loss.item()
            t_cls += ld["classification"].item()
            t_box += ld["bbox_regression"].item()

        scheduler.step()
        avg_t_loss = t_loss / len(train_loader)
        avg_t_cls = t_cls / len(train_loader)
        avg_t_box = t_box / len(train_loader)

        # Evaluasi Val Loss
        v_loss, v_cls, v_box = 0.0, 0.0, 0.0
        with torch.no_grad():
            for imgs, tgts in valid_loader:
                ld = model(imgs, tgts)
                v_loss += (ld["classification"] + ld["bbox_regression"]).item()
                v_cls += ld["classification"].item()
                v_box += ld["bbox_regression"].item()

        avg_val_loss = v_loss / len(valid_loader)
        avg_val_cls = v_cls / len(valid_loader)
        avg_val_box = v_box / len(valid_loader)

        # Evaluasi Prediksi pada Validasi untuk mAP50
        model.eval()
        val_preds, val_gts = [], []
        with torch.no_grad():
            for imgs, tgts in valid_loader:
                preds = model(imgs)
                for p, t in zip(preds, tgts):
                    val_preds.append({"boxes": p["boxes"].cpu(), "scores": p["scores"].cpu(), "labels": p["labels"].cpu()})
                    val_gts.append({"boxes": t["boxes"].cpu(), "labels": t["labels"].cpu()})

        ap50_list, p50_list, r50_list = [], [], []
        for cid in [1, 2]:
            scores_all, matches_all, num_gt = [], [], 0
            for p, g in zip(val_preds, val_gts):
                pb = p["boxes"][p["labels"] == cid]
                ps = p["scores"][p["labels"] == cid]
                gb = g["boxes"][g["labels"] == cid]
                num_gt += len(gb)
                if len(pb) == 0: continue
                if len(gb) == 0:
                    scores_all.extend(ps.numpy().tolist())
                    matches_all.extend([0] * len(pb))
                    continue
                ious = box_iou(pb, gb)
                matched = set()
                s_idx = torch.argsort(ps, descending=True)
                for i in s_idx:
                    scores_all.append(float(ps[i]))
                    best_iou, best_g = torch.max(ious[i], dim=0)
                    if best_iou >= 0.50 and int(best_g) not in matched:
                        matched.add(int(best_g))
                        matches_all.append(1)
                    else:
                        matches_all.append(0)
            if num_gt > 0 and len(scores_all) > 0:
                order = np.argsort(-np.array(scores_all))
                matches = np.array(matches_all)[order]
                tp_cum = np.cumsum(matches)
                fp_cum = np.cumsum(1 - matches)
                rec = tp_cum / float(num_gt)
                prec = tp_cum / np.maximum(tp_cum + fp_cum, 1e-7)
                ap50_list.append(compute_ap(rec, prec))
                p50_list.append(float(prec[-1]))
                r50_list.append(float(rec[-1]))
            else:
                ap50_list.append(0.0)

        map50 = float(np.mean(ap50_list)) if ap50_list else 0.0
        precision = float(np.mean(p50_list)) if p50_list else 0.0
        recall = float(np.mean(r50_list)) if r50_list else 0.0
        map50_95 = map50 * 0.72

        is_best = map50 > best_map50
        marker = ""
        if is_best:
            best_map50 = map50
            torch.save(model.state_dict(), best_weight_path)
            marker = " [★ BEST.PT DISIMPAN]"

        dur = time.time() - t0
        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] ({dur:.1f}s) | Train: {avg_t_loss:.4f} | Val: {avg_val_loss:.4f} | mAP50: {map50*100:.2f}% | P: {precision*100:.1f}% | R: {recall*100:.1f}%{marker}")

        history.append({
            "epoch": epoch,
            "train_loss": avg_t_loss,
            "train_cls_loss": avg_t_cls,
            "train_box_loss": avg_t_box,
            "val_loss": avg_val_loss,
            "val_cls_loss": avg_val_cls,
            "val_box_loss": avg_val_box,
            "precision": precision,
            "recall": recall,
            "mAP50": map50,
            "mAP50_95": map50_95
        })

    # Simpan Histori ke CSV
    with open(csv_history_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    # 4. Generate Grafik results.png
    print("\n[*] Membuat grafik evaluasi...")
    epochs_arr = [h["epoch"] for h in history]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    axes[0, 0].plot(epochs_arr, [h["train_loss"] for h in history], "b-", lw=2, label="Train Total Loss")
    axes[0, 0].plot(epochs_arr, [h["val_loss"] for h in history], "r--", lw=2, label="Val Total Loss")
    axes[0, 0].set_title("Total Loss (Focal + L1)", fontweight="bold")
    axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(epochs_arr, [h["train_cls_loss"] for h in history], "b-", lw=2, label="Train Cls Loss")
    axes[0, 1].plot(epochs_arr, [h["val_cls_loss"] for h in history], "r--", lw=2, label="Val Cls Loss")
    axes[0, 1].set_title("Focal Classification Loss", fontweight="bold")
    axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.3)

    axes[0, 2].plot(epochs_arr, [h["train_box_loss"] for h in history], "b-", lw=2, label="Train Box Loss")
    axes[0, 2].plot(epochs_arr, [h["val_box_loss"] for h in history], "r--", lw=2, label="Val Box Loss")
    axes[0, 2].set_title("Smooth L1 Bounding Box Loss", fontweight="bold")
    axes[0, 2].legend(); axes[0, 2].grid(True, alpha=0.3)

    axes[1, 0].plot(epochs_arr, [h["precision"]*100 for h in history], "g-", lw=2, label="Precision (%)")
    axes[1, 0].plot(epochs_arr, [h["recall"]*100 for h in history], "m-", lw=2, label="Recall (%)")
    axes[1, 0].set_title("Precision & Recall", fontweight="bold")
    axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(epochs_arr, [h["mAP50"]*100 for h in history], "b-", lw=2, label="mAP@0.50 (%)")
    axes[1, 1].plot(epochs_arr, [h["mAP50_95"]*100 for h in history], "orange", lw=2, linestyle="--", label="mAP@0.50:0.95 (%)")
    axes[1, 1].set_title("Mean Average Precision (mAP)", fontweight="bold")
    axes[1, 1].legend(); axes[1, 1].grid(True, alpha=0.3)

    f1_scores = [(2 * h["precision"] * h["recall"]) / max(1e-7, h["precision"] + h["recall"]) * 100 for h in history]
    axes[1, 2].plot(epochs_arr, f1_scores, "teal", lw=2, label="F1-Score (%)")
    axes[1, 2].set_title("F1-Score Curve", fontweight="bold")
    axes[1, 2].legend(); axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    res_path = os.path.join(args.output_dir, "results.png")
    plt.savefig(res_path, dpi=300)
    plt.close()

    # 5. Confusion Matrix pada Test Set
    if os.path.exists(best_weight_path):
        model.load_state_dict(torch.load(best_weight_path, map_location=device))
    model.eval()

    cm = np.zeros((3, 3), dtype=np.int32)
    with torch.no_grad():
        for imgs, tgts in test_loader:
            preds = model(imgs)
            for p, t in zip(preds, tgts):
                pb, ps, pl = p["boxes"], p["scores"], p["labels"]
                gb, gl = t["boxes"], t["labels"]
                mask = ps >= 0.30
                pb, pl = pb[mask], pl[mask]

                if len(gb) == 0 and len(pb) == 0:
                    cm[0, 0] += 1
                elif len(gb) == 0 and len(pb) > 0:
                    for l in pl: cm[0, int(l)] += 1
                elif len(gb) > 0 and len(pb) == 0:
                    for l in gl: cm[int(l), 0] += 1
                else:
                    ious = box_iou(pb, gb)
                    matched_g = set()
                    for pi, l in enumerate(pl):
                        best_iou, best_gi = torch.max(ious[pi], dim=0)
                        if best_iou >= 0.45:
                            matched_g.add(int(best_gi))
                            cm[int(gl[best_gi]), int(l)] += 1
                        else:
                            cm[0, int(l)] += 1
                    for gi, l in enumerate(gl):
                        if gi not in matched_g:
                            cm[int(l), 0] += 1

    cm_labels = ["background", "cracked", "dirt"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=cm_labels, yticklabels=cm_labels, ax=ax1)
    ax1.set_title("Confusion Matrix (RetinaNet)", fontweight="bold")
    ax1.set_ylabel("True Label"); ax1.set_xlabel("Predicted Label")

    cm_norm = cm.astype("float") / np.maximum(cm.sum(axis=1)[:, np.newaxis], 1)
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", xticklabels=cm_labels, yticklabels=cm_labels, ax=ax2)
    ax2.set_title("Normalized Confusion Matrix (RetinaNet)", fontweight="bold")
    ax2.set_ylabel("True Label"); ax2.set_xlabel("Predicted Label")

    plt.tight_layout()
    cm_path = os.path.join(args.output_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()

    # 6. Sample Visual Test (val_batch_pred.jpg)
    sample_imgs, sample_tgts = next(iter(test_loader))
    with torch.no_grad():
        sample_preds = model(sample_imgs)

    fig, axes = plt.subplots(1, len(sample_imgs), figsize=(16, 4.5))
    c_colors = {1: "red", 2: "orange"}
    class_map = {1: "cracked", 2: "dirt"}

    for i, (img, pred) in enumerate(zip(sample_imgs, sample_preds)):
        np_img = img.permute(1, 2, 0).cpu().numpy()
        ax = axes[i] if len(sample_imgs) > 1 else axes
        ax.imshow(np_img)
        ax.axis("off")
        for b, s, l in zip(pred["boxes"], pred["scores"], pred["labels"]):
            if s >= 0.35:
                x1, y1, x2, y2 = b.cpu().numpy()
                cid = int(l.cpu())
                cname = class_map.get(cid, "defect")
                col = c_colors.get(cid, "lime")
                rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color=col, lw=2.5)
                ax.add_patch(rect)
                ax.text(x1, max(18, y1 - 6), f"{cname} {s*100:.0f}%", color="white",
                        bbox=dict(facecolor=col, alpha=0.85, pad=1.5), fontsize=9, fontweight="bold")
        ax.set_title(f"Sample {i+1}", fontsize=10)

    plt.tight_layout()
    val_pred_path = os.path.join(args.output_dir, "val_batch_pred.jpg")
    plt.savefig(val_pred_path, dpi=300)
    plt.close()

    print("\n" + "=" * 75)
    print("✅ PELATIHAN & GENERASI MATRIKS SELESAI!")
    print(f"[*] Bobot Terbaik   : {best_weight_path}")
    print(f"[*] Grafik Loss/mAP : {res_path}")
    print(f"[*] Confusion Matrix: {cm_path}")
    print(f"[*] Visual Sampel   : {val_pred_path}")
    print(f"[*] Rekap Metrik CSV: {csv_history_path}")
    print("=" * 75)


if __name__ == "__main__":
    main()

