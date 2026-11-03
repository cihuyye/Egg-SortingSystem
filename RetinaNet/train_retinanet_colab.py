# ==============================================================================
# SCRIPT TRAINING RETINANET (RESNET-50 FPN V2) DI GOOGLE COLAB
# SISTEM SORTIR MUTU TELUR (SNI 3926:2023) - BENCHMARK VS YOLOV8
# ==============================================================================
# Petunjuk:
# 1. Buka Google Colab: https://colab.research.google.com
# 2. Ganti Runtime ke GPU (Runtime -> Change runtime type -> T4 GPU)
# 3. Upload file 'Egg_Dataset_COCO.zip' ke Colab (di tab files sebelah kiri)
# 4. Copy-paste dan jalankan kode di bawah ini
# ==============================================================================

import os
import zipfile
import json
import time
from typing import List, Dict, Any, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision
from torchvision.models.detection import retinanet_resnet50_fpn_v2, RetinaNet_ResNet50_FPN_V2_Weights
from torchvision.models.detection.retinanet import RetinaNetClassificationHead
import torchvision.transforms.functional as F

# ------------------------------------------------------------------------------
# 1. EKSTRAK DATASET COCO
# ------------------------------------------------------------------------------
ZIP_NAME = "Egg_Dataset_COCO.zip"
DATA_ROOT = "Egg_Dataset_COCO"

if not os.path.exists(DATA_ROOT):
    if os.path.exists(ZIP_NAME):
        print(f"📦 Mengekstrak {ZIP_NAME}...")
        with zipfile.ZipFile(ZIP_NAME, "r") as z:
            z.extractall(DATA_ROOT)
        print("✅ Ekstraksi selesai!")
    else:
        raise FileNotFoundError(f"File {ZIP_NAME} tidak ditemukan. Harap upload ke Google Colab!")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Perangkat yang digunakan: {DEVICE}")
if torch.cuda.is_available():
    print(f"🔥 GPU Model: {torch.cuda.get_device_name(0)}")

# ------------------------------------------------------------------------------
# 2. PYTORCH DATASET COCO
# ------------------------------------------------------------------------------
class EggCocoDataset(Dataset):
    def __init__(self, img_dir: str, json_path: str, is_train: bool = True):
        self.img_dir = img_dir
        self.is_train = is_train
        
        with open(json_path, "r") as f:
            self.coco = json.load(f)
            
        self.images = self.coco["images"]
        self.img_id_to_anns = {}
        for ann in self.coco["annotations"]:
            img_id = ann["image_id"]
            if img_id not in self.img_id_to_anns:
                self.img_id_to_anns[img_id] = []
            self.img_id_to_anns[img_id].append(ann)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        img_info = self.images[idx]
        img_path = os.path.join(self.img_dir, img_info["file_name"])
        img = Image.open(img_path).convert("RGB")
        
        anns = self.img_id_to_anns.get(img_info["id"], [])
        boxes = []
        labels = []
        areas = []
        iscrowd = []
        
        for ann in anns:
            # COCO bbox: [xmin, ymin, w, h] -> PyTorch format: [xmin, ymin, xmax, ymax]
            xmin, ymin, w, h = ann["bbox"]
            xmax = xmin + w
            ymax = ymin + h
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(ann["category_id"])  # 1: cracked, 2: dirt, 3: good
            areas.append(ann.get("area", w * h))
            iscrowd.append(ann.get("iscrowd", 0))

        if len(boxes) == 0:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)
            areas_tensor = torch.zeros((0,), dtype=torch.float32)
        else:
            boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32)
            labels_tensor = torch.as_tensor(labels, dtype=torch.int64)
            areas_tensor = torch.as_tensor(areas, dtype=torch.float32)

        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": torch.tensor([img_info["id"]]),
            "area": areas_tensor,
            "iscrowd": torch.as_tensor(iscrowd, dtype=torch.int64)
        }

        # Convert Image to Tensor
        img_tensor = F.to_tensor(img)
        return img_tensor, target

def collate_fn(batch):
    return tuple(zip(*batch))

# Buat DataLoader
train_dataset = EggCocoDataset(
    img_dir=os.path.join(DATA_ROOT, "train"),
    json_path=os.path.join(DATA_ROOT, "annotations", "instances_train.json"),
    is_train=True
)
val_dataset = EggCocoDataset(
    img_dir=os.path.join(DATA_ROOT, "val"),
    json_path=os.path.join(DATA_ROOT, "annotations", "instances_val.json"),
    is_train=False
)
test_dataset = EggCocoDataset(
    img_dir=os.path.join(DATA_ROOT, "test"),
    json_path=os.path.join(DATA_ROOT, "annotations", "instances_test.json"),
    is_train=False
)

# Batch size 8 ideal untuk RetinaNet ResNet-50 di GPU Colab T4
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

print(f"📊 Dataset siap: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")

# ------------------------------------------------------------------------------
# 3. INISIALISASI MODEL RETINANET (RESNET-50 FPN V2)
# ------------------------------------------------------------------------------
print("🧠 Memuat pre-trained RetinaNet ResNet-50 FPN V2...")
weights = RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT
model = retinanet_resnet50_fpn_v2(weights=weights)

# Sesuaikan Classification Head untuk 3 Kelas Telur (+ 1 latar belakang = 4 kelas total)
NUM_CLASSES = 4
num_anchors = model.head.classification_head.num_anchors
model.head.classification_head = RetinaNetClassificationHead(
    in_channels=256,
    num_anchors=num_anchors,
    num_classes=NUM_CLASSES
)
model.to(DEVICE)

# Hitung jumlah parameter
total_params = sum(p.numel() for p in model.parameters()) / 1e6
print(f"📦 Total Parameter RetinaNet: {total_params:.2f} Juta (Bandingkan dg YOLOv8n: ~3.2 Juta)")

# ------------------------------------------------------------------------------
# 4. OPTIMIZER & LEARNING RATE SCHEDULER
# ------------------------------------------------------------------------------
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=1e-4)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)

# ------------------------------------------------------------------------------
# 5. FUNGSI EVALUASI SEDERHANA (mAP & PRECISION / RECALL ESTIMATION)
# ------------------------------------------------------------------------------
def evaluate_model(model, data_loader, device, iou_thresh=0.5):
    model.eval()
    total_tp = 0
    total_fp = 0
    total_gt = 0
    inference_times = []

    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            
            t0 = time.time()
            predictions = model(images)
            inference_times.append((time.time() - t0) / len(images))

            for pred, target in zip(predictions, targets):
                p_boxes = pred["boxes"].cpu()
                p_scores = pred["scores"].cpu()
                p_labels = pred["labels"].cpu()

                gt_boxes = target["boxes"].cpu()
                gt_labels = target["labels"].cpu()
                total_gt += len(gt_labels)

                # Ambil deteksi dengan score > 0.4
                keep = p_scores > 0.4
                p_boxes = p_boxes[keep]
                p_labels = p_labels[keep]

                matched_gt = set()
                for pb, pl in zip(p_boxes, p_labels):
                    best_iou = 0.0
                    best_gt_idx = -1
                    for g_idx, (gb, gl) in enumerate(zip(gt_boxes, gt_labels)):
                        if g_idx in matched_gt or pl != gl:
                            continue
                        # Hitung IoU
                        ixmin = max(pb[0], gb[0])
                        iymin = max(pb[1], gb[1])
                        ixmax = min(pb[2], gb[2])
                        iymax = min(pb[3], gb[3])
                        iw = max(0.0, ixmax - ixmin)
                        ih = max(0.0, iymax - iymin)
                        inter = iw * ih
                        area_p = (pb[2] - pb[0]) * (pb[3] - pb[1])
                        area_g = (gb[2] - gb[0]) * (gb[3] - gb[1])
                        union = area_p + area_g - inter
                        iou = inter / union if union > 0 else 0.0
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = g_idx

                    if best_iou >= iou_thresh and best_gt_idx >= 0:
                        total_tp += 1
                        matched_gt.add(best_gt_idx)
                    else:
                        total_fp += 1

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / total_gt if total_gt > 0 else 0.0
    avg_latency_ms = (sum(inference_times) / len(inference_times)) * 1000.0 if inference_times else 0.0

    return precision, recall, avg_latency_ms

# ------------------------------------------------------------------------------
# 6. TRAINING LOOP (50 EPOCH)
# ------------------------------------------------------------------------------
EPOCHS = 50
best_recall = 0.0
SAVE_PATH = "best_retinanet.pth"

print(f"\n🚀 Memulai Pelatihan RetinaNet ({EPOCHS} Epoch)...")
print("=" * 70)

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0
    start_epoch = time.time()

    for images, targets in train_loader:
        images = [img.to(DEVICE) for img in images]
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += losses.item()

    lr_scheduler.step()
    epoch_time = time.time() - start_epoch
    avg_loss = total_loss / len(train_loader)

    # Evaluasi setiap 5 epoch dan di epoch terakhir
    if epoch % 5 == 0 or epoch == EPOCHS:
        prec, rec, lat = evaluate_model(model, val_loader, DEVICE)
        print(f"Epoch [{epoch:02d}/{EPOCHS}] - Loss: {avg_loss:.4f} | Val Prec: {prec*100:.2f}% | Val Recall: {rec*100:.2f}% | Time: {epoch_time:.1f}s")
        if rec >= best_recall:
            best_recall = rec
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"  ⭐ Model terbaik disimpan ke '{SAVE_PATH}' (Recall: {rec*100:.2f}%)")
    else:
        print(f"Epoch [{epoch:02d}/{EPOCHS}] - Loss: {avg_loss:.4f} | Time: {epoch_time:.1f}s")

print("=" * 70)
print(f"🎉 Pelatihan Selesai! Model terbaik tersimpan di: {SAVE_PATH}\n")

# ------------------------------------------------------------------------------
# 7. EVALUASI AKHIR PADA TEST SET (HASIL PERBANDINGAN)
# ------------------------------------------------------------------------------
print("🏁 Menjalankan Pengujian Akhir pada Test Set (80 Citra)...")
model.load_state_dict(torch.load(SAVE_PATH))
test_prec, test_rec, test_lat = evaluate_model(model, test_loader, DEVICE)
model_size_mb = os.path.getsize(SAVE_PATH) / (1024 * 1024)

print("\n" + "=" * 65)
print("📊 HASIL AKHIR RETINANET (RESNET-50 FPN V2)")
print("=" * 65)
print(f"• Test Precision   : {test_prec * 100:.2f}%")
print(f"• Test Recall      : {test_rec * 100:.2f}%")
print(f"• Rata-rata Latensi: {test_lat:.2f} ms / citra (GPU)")
print(f"• Ukuran Model     : {model_size_mb:.2f} MB")
print(f"• Jumlah Parameter : {total_params:.2f} Juta")
print("=" * 65)

