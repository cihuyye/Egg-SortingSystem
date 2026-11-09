"""
Dataset loader for RetinaNet using YOLOv8 annotation format (polygon & bbox).
Mendukung format label YOLOv8 (Polygon segmentation & Bounding Box), serta
negative background samples (good eggs / file label kosong).
"""

import os
import glob
import random
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF

CLASS_MAPPING_YOLO = {
    0: "cracked",
    1: "dirt"
}

# RetinaNet Convention:
# 0: background
# 1: cracked
# 2: dirt
CLASS_NAMES = ["__background__", "cracked", "dirt"]
NUM_CLASSES = len(CLASS_NAMES)


class EggYoloRetinaDataset(Dataset):
    """
    Dataset loader yang membaca langsung dari struktur dataset YOLOv8:
    dataset_dir/
      ├── train/
      │   ├── images/ (cracked, dirt, good)
      │   └── labels/ (cracked, dirt, good)
      ├── valid/
      └── test/
    """
    def __init__(self, root_dir, split="train", imgsz=640, augment=True):
        self.root_dir = root_dir
        self.split = split
        self.imgsz = imgsz
        self.augment = augment and (split == "train")

        self.images_dir = os.path.join(root_dir, split, "images")
        self.labels_dir = os.path.join(root_dir, split, "labels")

        # Kumpulkan semua file citra .jpg, .jpeg, .png
        valid_exts = (".jpg", ".jpeg", ".png")
        self.image_paths = []
        for ext in valid_exts:
            pattern = os.path.join(self.images_dir, "**", f"*{ext}")
            self.image_paths.extend(glob.glob(pattern, recursive=True))

        self.image_paths.sort()
        if len(self.image_paths) == 0:
            raise FileNotFoundError(f"Tidak ada citra ditemukan di {self.images_dir}")

    def __len__(self):
        return len(self.image_paths)

    def _find_label_path(self, img_path):
        """Mencari file label .txt yang sesuai dengan path citra"""
        # Coba cara 1: Ganti folder images dengan labels
        lbl_path = img_path.replace(
            f"{os.path.sep}images{os.path.sep}",
            f"{os.path.sep}labels{os.path.sep}"
        )
        lbl_path = os.path.splitext(lbl_path)[0] + ".txt"
        if os.path.exists(lbl_path):
            return lbl_path

        # Coba cara 2: Cari di seluruh subfolder labels berdasarkan basename
        base_name = os.path.splitext(os.path.basename(img_path))[0] + ".txt"
        match = glob.glob(os.path.join(self.labels_dir, "**", base_name), recursive=True)
        if match:
            return match[0]

        return None

    def _parse_label_file(self, lbl_path, orig_w, orig_h):
        """
        Membaca label YOLO:
        - Jika file kosong: negative sample (good egg)
        - Jika len(coords) == 4: standard bbox (cx, cy, w, h)
        - Jika len(coords) > 4: polygon coordinates (x1, y1, x2, y2, ...)
        Mengembalikan boxes [xmin, ymin, xmax, ymax] dan labels (1-indexed).
        """
        boxes = []
        labels = []

        if lbl_path is None or not os.path.exists(lbl_path):
            return boxes, labels

        with open(lbl_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        for line in lines:
            parts = line.split()
            if len(parts) < 5:
                continue

            yolo_cls = int(float(parts[0]))
            coords = [float(x) for x in parts[1:]]

            # RetinaNet convention: 0 adalah background, sehingga kelas bergeser +1
            retina_cls = yolo_cls + 1

            if len(coords) == 4:
                # Format Standar Bounding Box YOLO (center_x, center_y, width, height)
                cx, cy, bw, bh = coords
                xmin = (cx - bw / 2.0) * orig_w
                ymin = (cy - bh / 2.0) * orig_h
                xmax = (cx + bw / 2.0) * orig_w
                ymax = (cy + bh / 2.0) * orig_h
            else:
                # Format Poligon Segmen YOLO (x1, y1, x2, y2, ...)
                xs = coords[0::2]
                ys = coords[1::2]
                xmin = min(xs) * orig_w
                ymin = min(ys) * orig_h
                xmax = max(xs) * orig_w
                ymax = max(ys) * orig_h

            # Kliping koordinat ke batas citra
            xmin = max(0.0, min(xmin, float(orig_w)))
            ymin = max(0.0, min(ymin, float(orig_h)))
            xmax = max(0.0, min(xmax, float(orig_w)))
            ymax = max(0.0, min(ymax, float(orig_h)))

            # Validasi ukuran box minimal 2x2 piksel
            if (xmax - xmin) >= 2.0 and (ymax - ymin) >= 2.0:
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(retina_cls)

        return boxes, labels

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        orig_w, orig_h = image.size

        lbl_path = self._find_label_path(img_path)
        boxes, labels = self._parse_label_file(lbl_path, orig_w, orig_h)

        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        # 1. Resize citra dan skala koordinat box
        target_w = self.imgsz
        target_h = self.imgsz
        if orig_w != target_w or orig_h != target_h:
            image = image.resize((target_w, target_h), Image.BILINEAR)
            if len(boxes) > 0:
                scale_x = target_w / orig_w
                scale_y = target_h / orig_h
                boxes[:, [0, 2]] *= scale_x
                boxes[:, [1, 3]] *= scale_y

        # 2. Augmentasi data (hanya pada split train)
        if self.augment:
            # Horizontal Flip (Probabilitas 50%)
            if random.random() > 0.5:
                image = TF.hflip(image)
                if len(boxes) > 0:
                    old_xmin = boxes[:, 0].clone()
                    old_xmax = boxes[:, 2].clone()
                    boxes[:, 0] = target_w - old_xmax
                    boxes[:, 2] = target_w - old_xmin

            # Color Jitter (Kecerahan & Kontras Ringan)
            if random.random() > 0.5:
                brightness_factor = random.uniform(0.8, 1.2)
                image = TF.adjust_brightness(image, brightness_factor)
                contrast_factor = random.uniform(0.8, 1.2)
                image = TF.adjust_contrast(image, contrast_factor)

        # 3. Konversi ke Float Tensor [0.0 - 1.0]
        img_tensor = TF.to_tensor(image)

        # 4. Siapkan target dictionary sesuai standar Torchvision Detection Models
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([idx])
        if len(boxes) > 0:
            target["area"] = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            target["iscrowd"] = torch.zeros((len(boxes),), dtype=torch.int64)
        else:
            target["area"] = torch.zeros((0,), dtype=torch.float32)
            target["iscrowd"] = torch.zeros((0,), dtype=torch.int64)

        return img_tensor, target


def collate_fn(batch):
    """
    Collate function khusus untuk object detection Torchvision:
    Menggabungkan list of (image, target) menjadi (tuple of images, tuple of targets).
    """
    return tuple(zip(*batch))

