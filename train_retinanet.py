"""
Script Pelatihan Model RetinaNet (ResNet-50 FPN) untuk Deteksi Cacat Telur (Cracked & Dirt).
Membaca langsung dari folder dataset berformat YOLOv8 ('Egg Dataset Clean').

Contoh Penggunaan:
    python train_retinanet.py --epochs 50 --batch-size 8 --lr 0.0001
"""

import os
import sys
import time
import argparse
import csv
import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from retinanet_dataset import EggYoloRetinaDataset, collate_fn, NUM_CLASSES
from retinanet_model import create_retinanet_model, save_checkpoint
from retinanet_metrics import evaluate_detections


def parse_args():
    parser = argparse.ArgumentParser(description="Pelatihan Model RetinaNet ResNet-50 FPN")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="Egg Dataset Clean",
        help="Path ke folder dataset (default: 'Egg Dataset Clean')"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Jumlah epoch pelatihan (default: 50)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Ukuran batch per iterasi (default: 8, rekomendasi GPU 4-8GB: 4-8, Colab T4: 8-16)"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Resolusi citra input (default: 640)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate awal (default: 1e-4 untuk AdamW, 1e-3 untuk SGD)"
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay regularisasi (default: 1e-4)"
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        choices=["adamw", "sgd"],
        default="adamw",
        help="Optimizer (default: adamw)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="runs/retinanet",
        help="Direktori penyimpanan hasil dan bobot model (default: runs/retinanet)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Perangkat pelatihan ('cuda' atau 'cpu')"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2 if os.name != "nt" else 0,
        help="Jumlah worker DataLoader (default: 0 di Windows untuk hindari issue multiprocessing)"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=15,
        help="Early stopping patience (default: 15 epoch tanpa peningkatan mAP50)"
    )
    parser.add_argument(
        "--version",
        type=str,
        default="v1",
        choices=["v1", "v2"],
        help="Versi arsitektur RetinaNet (default: v1)"
    )
    return parser.parse_args()


def plot_training_curves(history, save_path):
    """Membuat grafik loss dan mAP dari log pelatihan"""
    epochs = [h["epoch"] for h in history]
    train_losses = [h["train_loss"] for h in history]
    val_losses = [h["val_loss"] for h in history]
    map50s = [h["mAP50"] * 100 for h in history]
    map50_95s = [h["mAP50_95"] * 100 for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Grafik Loss
    ax1.plot(epochs, train_losses, label="Train Total Loss", color="#1f77b4", lw=2)
    ax1.plot(epochs, val_losses, label="Val Total Loss", color="#ff7f0e", lw=2, linestyle="--")
    ax1.set_title("Grafik Loss Pelatihan & Validasi", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Grafik mAP
    ax2.plot(epochs, map50s, label="mAP @ 0.50 (%)", color="#2ca02c", lw=2)
    ax2.plot(epochs, map50_95s, label="mAP @ 0.50:0.95 (%)", color="#9467bd", lw=2, linestyle="--")
    ax2.set_title("Metrik Evaluasi mAP pada Set Validasi", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("mAP (%)")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def train():
    args = parse_args()
    device = torch.device(args.device)

    print("=" * 70)
    print("🥚 PELATIHAN MODEL RETINANET (RESNET-50 FPN) - DETEKSI CACAT TELUR")
    print("=" * 70)
    print(f"[*] Perangkat: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print(f"[*] Dataset Root: {args.dataset_dir}")
    print(f"[*] Epochs: {args.epochs} | Batch Size: {args.batch_size} | Resolusi: {args.imgsz}x{args.imgsz}")
    print(f"[*] Optimizer: {args.optimizer.upper()} | Initial LR: {args.lr}")
    print(f"[*] Output Dir: {args.output_dir}")
    print("=" * 70)

    # 1. Inisialisasi Dataset & DataLoader
    train_dataset = EggYoloRetinaDataset(
        root_dir=args.dataset_dir,
        split="train",
        imgsz=args.imgsz,
        augment=True
    )
    val_dataset = EggYoloRetinaDataset(
        root_dir=args.dataset_dir,
        split="valid",
        imgsz=args.imgsz,
        augment=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda")
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda")
    )

    print(f"[*] Citra Latih: {len(train_dataset)} citra ({len(train_loader)} batch)")
    print(f"[*] Citra Validasi: {len(val_dataset)} citra ({len(val_loader)} batch)")

    # 2. Bangun Model RetinaNet
    print("\n[*] Menginisialisasi RetinaNet dengan pre-trained weights COCO...")
    model = create_retinanet_model(
        num_classes=NUM_CLASSES,  # 3 kelas: background, cracked, dirt
        pretrained=True,
        version=args.version
    )
    model.to(device)

    # 3. Setup Optimizer & Learning Rate Scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    if args.optimizer.lower() == "adamw":
        optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

    # Cosine Annealing Scheduler dengan minimum LR 1e-6
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6
    )

    # Mixed Precision Scaler untuk akselerasi GPU
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # Folder output
    os.makedirs(args.output_dir, exist_ok=True)
    best_weight_path = os.path.join(args.output_dir, "best_retinanet.pth")
    last_weight_path = os.path.join(args.output_dir, "last_retinanet.pth")
    history_csv_path = os.path.join(args.output_dir, "training_history.csv")
    curve_plot_path = os.path.join(args.output_dir, "loss_curves.png")

    best_map50 = 0.0
    epochs_no_improve = 0
    history = []

    print("\n[*] Memulai proses training loop...")
    print("-" * 70)

    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()

        # ==========================================
        # 1. TRAINING PHASE
        # ==========================================
        model.train()
        train_loss_total = 0.0
        train_cls_loss = 0.0
        train_box_loss = 0.0
        num_batches = 0

        for images, targets in train_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                loss_dict = model(images, targets)
                cls_loss = loss_dict["classification"]
                box_loss = loss_dict["bbox_regression"]
                losses = cls_loss + box_loss

            scaler.scale(losses).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss_total += losses.item()
            train_cls_loss += cls_loss.item()
            train_box_loss += box_loss.item()
            num_batches += 1

        avg_train_loss = train_loss_total / max(1, num_batches)
        avg_train_cls = train_cls_loss / max(1, num_batches)
        avg_train_box = train_box_loss / max(1, num_batches)

        # Update Learning Rate
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        # ==========================================
        # 2. VALIDATION LOSS PHASE
        # ==========================================
        # Model tetap dalam mode .train() untuk menghitung nilai loss validasi tanpa gradien
        val_loss_total = 0.0
        val_batches = 0
        with torch.no_grad():
            for val_imgs, val_targets in val_loader:
                val_imgs = [img.to(device) for img in val_imgs]
                val_targets = [{k: v.to(device) for k, v in t.items()} for t in val_targets]
                with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                    v_loss_dict = model(val_imgs, val_targets)
                    v_loss = v_loss_dict["classification"] + v_loss_dict["bbox_regression"]
                val_loss_total += v_loss.item()
                val_batches += 1

        avg_val_loss = val_loss_total / max(1, val_batches)

        # ==========================================
        # 3. EVALUASI METRIK DETEKSI (mAP)
        # ==========================================
        model.eval()
        all_preds = []
        all_gt = []

        with torch.no_grad():
            for val_imgs, val_targets in val_loader:
                val_imgs = [img.to(device) for img in val_imgs]
                preds = model(val_imgs)

                for p, t in zip(preds, val_targets):
                    all_preds.append({
                        "boxes": p["boxes"].cpu(),
                        "scores": p["scores"].cpu(),
                        "labels": p["labels"].cpu()
                    })
                    all_gt.append({
                        "boxes": t["boxes"].cpu(),
                        "labels": t["labels"].cpu()
                    })

        metrics = evaluate_detections(all_preds, all_gt, class_ids=(1, 2))
        map50 = metrics["mAP50"]
        map50_95 = metrics["mAP50_95"]
        precision = metrics["precision"]
        recall = metrics["recall"]

        epoch_duration = time.time() - epoch_start_time

        # Simpan checkpoint terakhir
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            metrics=metrics,
            save_path=last_weight_path
        )

        # Cek performa terbaik (berdasarkan mAP50)
        is_best = map50 > best_map50
        if is_best:
            best_map50 = map50
            epochs_no_improve = 0
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                metrics=metrics,
                save_path=best_weight_path
            )
            marker = " ★ BEST"
        else:
            epochs_no_improve += 1
            marker = ""

        # Log ke terminal
        print(
            f"Epoch [{epoch:02d}/{args.epochs:02d}] ({epoch_duration:.1f}s) | "
            f"Train: {avg_train_loss:.4f} (cls: {avg_train_cls:.4f}, box: {avg_train_box:.4f}) | "
            f"Val: {avg_val_loss:.4f} | "
            f"mAP50: {map50 * 100:.2f}% | mAP50-95: {map50_95 * 100:.2f}% | "
            f"P: {precision * 100:.1f}% | R: {recall * 100:.1f}%{marker}"
        )

        # Catat histori ke list & CSV
        row = {
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 5),
            "train_cls_loss": round(avg_train_cls, 5),
            "train_box_loss": round(avg_train_box, 5),
            "val_loss": round(avg_val_loss, 5),
            "mAP50": round(map50, 5),
            "mAP50_95": round(map50_95, 5),
            "precision": round(precision, 5),
            "recall": round(recall, 5),
            "lr": round(current_lr, 7)
        }
        history.append(row)

        with open(history_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerows(history)

        # Update grafik plot kurva loss
        plot_training_curves(history, curve_plot_path)

        # Early stopping check
        if epochs_no_improve >= args.patience:
            print(f"\n[!] Early stopping terpicu: Tidak ada peningkatan mAP50 selama {args.patience} epoch berturut-turut.")
            break

    print("\n" + "=" * 70)
    print(f"✅ PELATIHAN RETINANET SELESAI!")
    print(f"[*] Bobot Terbaik : {best_weight_path} (Best mAP50: {best_map50 * 100:.2f}%)")
    print(f"[*] Histori CSV   : {history_csv_path}")
    print(f"[*] Grafik Plot   : {curve_plot_path}")
    print("=" * 70)


if __name__ == "__main__":
    train()

