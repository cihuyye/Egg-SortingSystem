"""
================================================================================
SKRIP KONVERSI DATASET DETEKSI OBJEK: FORMAT YOLO KE FORMAT MS-COCO JSON
Proyek: Sistem Klasifikasi & Sortir Mutu Telur Otomatis (SNI 3926:2023)
================================================================================

Tujuan Akademik:
Mengonversi dataset dari format terdesentralisasi YOLO (.txt dengan koordinat ternormalisasi)
ke format standar terpusat MS-COCO (.json dengan koordinat piksel absolut) untuk
melatih model pembanding RetinaNet (ResNet-50 FPN).

Formulasi Transformasi Bounding Box:
1. YOLO Format (Ternormalisasi 0.0 - 1.0):
   - [class_id, x_center_norm, y_center_norm, width_norm, height_norm]

2. MS-COCO Format (Piksel Absolut):
   - width_px  = width_norm * image_width
   - height_px = height_norm * image_height
   - x_min_px  = (x_center_norm * image_width) - (width_px / 2)
   - y_min_px  = (y_center_norm * image_height) - (height_px / 2)
   - area      = width_px * height_px
================================================================================
"""

import os
import json
import shutil
from typing import List, Dict, Any


def convert_yolo_to_coco(
    src_dataset_dir: str = None,
    dst_dataset_dir: str = None,
    img_width: int = 640,
    img_height: int = 640
) -> None:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_root = os.path.dirname(base_dir)

    if src_dataset_dir is None:
        src_dataset_dir = os.path.join(workspace_root, "Egg Dataset")
    if dst_dataset_dir is None:
        dst_dataset_dir = os.path.join(workspace_root, "Egg Dataset COCO")

    # 1. Definisi Taksonomi Kategori Kelas Standar COCO (ID dimulai dari 1)
    categories: List[Dict[str, Any]] = [
        {"id": 1, "name": "cracked", "supercategory": "egg_defect"},
        {"id": 2, "name": "dirt", "supercategory": "egg_defect"},
        {"id": 3, "name": "good", "supercategory": "egg_defect"}
    ]

    # Pemetaan subset dataset (Train, Validation, Test)
    split_configs = [
        ("train", "train", "instances_train.json"),
        ("valid", "val", "instances_val.json"),
        ("test", "test", "instances_test.json")
    ]

    annotations_dir = os.path.join(dst_dataset_dir, "annotations")
    os.makedirs(annotations_dir, exist_ok=True)

    print("=" * 70)
    print("  MEMULAI KONVERSI FORMAT DATASET TELUR: YOLO (.TXT) -> COCO (.JSON)")
    print("=" * 70)
    print(f"Sumber Input : {src_dataset_dir}")
    print(f"Target Output: {dst_dataset_dir}\n")

    for split_in, split_out, json_name in split_configs:
        dst_img_dir = os.path.join(dst_dataset_dir, split_out)
        os.makedirs(dst_img_dir, exist_ok=True)

        coco_images: List[Dict[str, Any]] = []
        coco_annotations: List[Dict[str, Any]] = []
        
        image_id_counter = 1
        annotation_id_counter = 1

        for cat_name in ["cracked", "dirt", "good"]:
            src_img_folder = os.path.join(src_dataset_dir, split_in, "images", cat_name)
            src_lbl_folder = os.path.join(src_dataset_dir, split_in, "labels", cat_name)

            if not os.path.exists(src_img_folder):
                continue

            image_files = sorted([f for f in os.listdir(src_img_folder) if f.lower().endswith((".jpg", ".png", ".jpeg"))])

            for img_file in image_files:
                # 2. Salin citra ke direktori target yang seragam
                src_img_path = os.path.join(src_img_folder, img_file)
                dst_img_path = os.path.join(dst_img_dir, img_file)
                if not os.path.exists(dst_img_path):
                    shutil.copy2(src_img_path, dst_img_path)

                # 3. Entri Objek Citra COCO
                coco_images.append({
                    "id": image_id_counter,
                    "file_name": img_file,
                    "width": img_width,
                    "height": img_height
                })

                # 4. Baca dan transformasi file label YOLO (.txt)
                lbl_file = os.path.splitext(img_file)[0] + ".txt"
                lbl_path = os.path.join(src_lbl_folder, lbl_file)

                if os.path.exists(lbl_path):
                    with open(lbl_path, "r") as f:
                        lines = f.readlines()

                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            cls_yolo = int(parts[0])  # 0: cracked, 1: dirt, 2: good
                            cx_norm = float(parts[1])
                            cy_norm = float(parts[2])
                            w_norm = float(parts[3])
                            h_norm = float(parts[4])

                            # Konversi Koordinat Ternormalisasi ke Piksel Absolut (COCO)
                            w_px = w_norm * img_width
                            h_px = h_norm * img_height
                            x_min = max(0.0, (cx_norm * img_width) - (w_px / 2.0))
                            y_min = max(0.0, (cy_norm * img_height) - (h_px / 2.0))
                            box_area = w_px * h_px

                            coco_annotations.append({
                                "id": annotation_id_counter,
                                "image_id": image_id_counter,
                                "category_id": cls_yolo + 1,  # Offset +1 (0->1, 1->2, 2->3)
                                "bbox": [round(x_min, 2), round(y_min, 2), round(w_px, 2), round(h_px, 2)],
                                "area": round(box_area, 2),
                                "iscrowd": 0
                            })
                            annotation_id_counter += 1

                image_id_counter += 1

        # 5. Struktur Lengkap Dokumen MS-COCO
        coco_dataset = {
            "info": {
                "description": "Egg Defect Dataset (COCO Format) untuk Pelatihan RetinaNet",
                "version": "1.0",
                "year": 2026,
                "standard": "MS COCO 2017 Format"
            },
            "licenses": [],
            "categories": categories,
            "images": coco_images,
            "annotations": coco_annotations
        }

        output_json_path = os.path.join(annotations_dir, json_name)
        with open(output_json_path, "w", encoding="utf-8") as jf:
            json.dump(coco_dataset, jf, indent=2)

        print(f"✔ Berhasil memproses subset [{split_out.upper()}]:")
        print(f"  - Total Citra     : {len(coco_images)}")
        print(f"  - Total Anotasi   : {len(coco_annotations)}")
        print(f"  - File Anotasi    : {output_json_path}\n")

    print("=" * 70)
    print("  KONVERSI SELESAI DENGAN SUKSES!")
    print("=" * 70)


if __name__ == "__main__":
    convert_yolo_to_coco()

