"""
Konverter Dataset YOLOv8 (Format Poligon / BBox) ke Format Standar COCO JSON.
Menghasilkan file anotasi COCO:
  - Egg Dataset Clean/annotations/instances_train.json
  - Egg Dataset Clean/annotations/instances_valid.json
  - Egg Dataset Clean/annotations/instances_test.json

Berguna untuk benchmarking dengan Detectron2, MMDetection, atau Torchvision COCO API.
"""

import os
import glob
import json
from PIL import Image


def polygon_area(xs, ys):
    """Menghitung luas poligon menggunakan rumus Shoelace (Gauss)"""
    return 0.5 * abs(sum(xs[i] * ys[i + 1] - xs[i + 1] * ys[i] for i in range(-1, len(xs) - 1)))


def convert_yolo_to_coco(dataset_dir="Egg Dataset Clean", output_dir=None):
    if output_dir is None:
        output_dir = os.path.join(dataset_dir, "annotations")
    os.makedirs(output_dir, exist_ok=True)

    categories = [
        {"id": 1, "name": "cracked", "supercategory": "defect"},
        {"id": 2, "name": "dirt", "supercategory": "defect"}
    ]

    splits = ["train", "valid", "test"]

    print("=" * 70)
    print("🔄 KONVERSI DATASET YOLOV8 KE COCO JSON")
    print("=" * 70)

    for split in splits:
        images_dir = os.path.join(dataset_dir, split, "images")
        labels_dir = os.path.join(dataset_dir, split, "labels")

        valid_exts = (".jpg", ".jpeg", ".png")
        image_paths = []
        for ext in valid_exts:
            image_paths.extend(glob.glob(os.path.join(images_dir, "**", f"*{ext}"), recursive=True))

        image_paths.sort()
        print(f"[*] Memproses split '{split}': {len(image_paths)} citra...")

        coco_images = []
        coco_annotations = []
        annotation_id = 1

        for img_id, img_path in enumerate(image_paths, 1):
            with Image.open(img_path) as im:
                w, h = im.size

            rel_file_name = os.path.relpath(img_path, dataset_dir).replace("\\", "/")

            coco_images.append({
                "id": img_id,
                "file_name": rel_file_name,
                "width": w,
                "height": h
            })

            # Cari label .txt
            lbl_path = img_path.replace(f"{os.path.sep}images{os.path.sep}", f"{os.path.sep}labels{os.path.sep}")
            lbl_path = os.path.splitext(lbl_path)[0] + ".txt"

            if not os.path.exists(lbl_path):
                continue

            with open(lbl_path, "r", encoding="utf-8") as f:
                lines = [l.strip() for l in f if l.strip()]

            for line in lines:
                parts = line.split()
                if len(parts) < 5:
                    continue

                yolo_cls = int(float(parts[0]))
                category_id = yolo_cls + 1  # 1: cracked, 2: dirt
                coords = [float(x) for x in parts[1:]]

                if len(coords) == 4:
                    cx, cy, bw, bh = coords
                    xmin = (cx - bw / 2.0) * w
                    ymin = (cy - bh / 2.0) * h
                    bbox_w = bw * w
                    bbox_h = bh * h
                    area = bbox_w * bbox_h
                    segmentation = [
                        [xmin, ymin, xmin + bbox_w, ymin, xmin + bbox_w, ymin + bbox_h, xmin, ymin + bbox_h]
                    ]
                else:
                    xs = [coords[i] * w for i in range(0, len(coords), 2)]
                    ys = [coords[i] * h for i in range(1, len(coords), 2)]
                    xmin = min(xs)
                    ymin = min(ys)
                    xmax = max(xs)
                    ymax = max(ys)
                    bbox_w = xmax - xmin
                    bbox_h = ymax - ymin
                    area = polygon_area(xs, ys)

                    seg_pts = []
                    for x, y in zip(xs, ys):
                        seg_pts.extend([round(x, 2), round(y, 2)])
                    segmentation = [seg_pts]

                coco_annotations.append({
                    "id": annotation_id,
                    "image_id": img_id,
                    "category_id": category_id,
                    "bbox": [round(xmin, 2), round(ymin, 2), round(bbox_w, 2), round(bbox_h, 2)],
                    "area": round(area, 2),
                    "segmentation": segmentation,
                    "iscrowd": 0
                })
                annotation_id += 1

        coco_dict = {
            "info": {
                "description": "Egg Defect Dataset (Cracked & Dirt)",
                "version": "1.0",
                "year": 2026
            },
            "licenses": [],
            "images": coco_images,
            "annotations": coco_annotations,
            "categories": categories
        }

        out_json_path = os.path.join(output_dir, f"instances_{split}.json")
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(coco_dict, f, indent=2)

        print(f"    -> Berhasil disimpan: {out_json_path} ({len(coco_annotations)} anotasi)")

    print("=" * 70)
    print("✅ Konversi selesai seluruhnya!")
    print("=" * 70)


if __name__ == "__main__":
    convert_yolo_to_coco()

