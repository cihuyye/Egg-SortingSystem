"""
Script Inferensi & Visualisasi Deteksi Objek RetinaNet.
Mendeteksi cacat telur (cracked & dirt) pada satu citra atau satu folder citra,
serta menyediakan class 'RetinaNetDetector' yang siap diintegrasikan ke pipeline sistem sortir telur.

Contoh Penggunaan:
    python predict_retinanet.py --source "Egg Dataset Clean/test/images/cracked" --weights runs/retinanet/best_retinanet.pth
"""

import os
import glob
import argparse
import cv2
import numpy as np
import torch
from PIL import Image
import torchvision.transforms.functional as TF

from retinanet_model import load_checkpoint


COLOR_MAP = {
    "cracked": (0, 0, 230),  # Merah terang (BGR)
    "dirt": (0, 165, 255),    # Oranye / Kuning tua (BGR)
}


class RetinaNetDetector:
    """
    Kelas pendeteksi cacat telur berbasis RetinaNet ResNet-50 FPN.
    Dapat digunakan sebagai pengganti atau pembanding YOLOv8 di aplikasi sortir telur.
    """
    def __init__(
        self,
        weights_path="runs/retinanet/best_retinanet.pth",
        device="cuda" if torch.cuda.is_available() else "cpu",
        conf_thresh=0.40,
        imgsz=640
    ):
        self.device = torch.device(device)
        self.conf_thresh = conf_thresh
        self.imgsz = imgsz

        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Bobot RetinaNet tidak ditemukan di: {weights_path}")

        self.model, self.class_names = load_checkpoint(
            weights_path=weights_path,
            device=self.device,
            score_thresh=conf_thresh
        )
        self.idx_to_label = {1: "cracked", 2: "dirt"}

    def detect(self, image_input):
        """
        Mendeteksi cacat pada citra input (numpy BGR dari cv2.imread atau PIL Image atau path file string).

        Returns:
            dict: {
                'has_defect': bool,
                'defect_type': 'cracked' | 'dirt' | 'good',
                'detections': list of dict [{'class': str, 'conf': float, 'box': [x1, y1, x2, y2]}],
                'annotated_image': np.ndarray (BGR)
            }
        """
        # Konversi input menjadi PIL RGB dan numpy BGR
        if isinstance(image_input, str):
            orig_bgr = cv2.imread(image_input)
            if orig_bgr is None:
                raise ValueError(f"Gagal membaca citra dari {image_input}")
            pil_img = Image.fromarray(cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB))
        elif isinstance(image_input, np.ndarray):
            orig_bgr = image_input.copy()
            pil_img = Image.fromarray(cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB))
        elif isinstance(image_input, Image.Image):
            pil_img = image_input.convert("RGB")
            orig_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        else:
            raise TypeError("Format input harus berupa str path, numpy.ndarray (OpenCV), atau PIL.Image")

        orig_w, orig_h = pil_img.size

        # Resize untuk input model
        resized = pil_img.resize((self.imgsz, self.imgsz), Image.BILINEAR)
        img_tensor = TF.to_tensor(resized).unsqueeze(0).to(self.device)

        scale_x = orig_w / float(self.imgsz)
        scale_y = orig_h / float(self.imgsz)

        # Inferensi model
        with torch.no_grad():
            preds = self.model(img_tensor)[0]

        boxes = preds["boxes"].cpu()
        scores = preds["scores"].cpu()
        labels = preds["labels"].cpu()

        detections = []
        annotated = orig_bgr.copy()

        found_cracked = False
        found_dirt = False

        for b, s, l in zip(boxes, scores, labels):
            score_val = float(s)
            if score_val < self.conf_thresh:
                continue

            cls_id = int(l)
            cls_name = self.idx_to_label.get(cls_id, f"class_{cls_id}")

            if cls_name == "cracked":
                found_cracked = True
            elif cls_name == "dirt":
                found_dirt = True

            # Kembalikan koordinat ke ukuran asli
            x1 = int(round(float(b[0]) * scale_x))
            y1 = int(round(float(b[1]) * scale_y))
            x2 = int(round(float(b[2]) * scale_x))
            y2 = int(round(float(b[3]) * scale_y))

            # Batasi koordinat
            x1 = max(0, min(x1, orig_w - 1))
            y1 = max(0, min(y1, orig_h - 1))
            x2 = max(0, min(x2, orig_w - 1))
            y2 = max(0, min(y2, orig_h - 1))

            detections.append({
                "class": cls_name,
                "conf": round(score_val, 4),
                "box": [x1, y1, x2, y2]
            })

            # Gambar visualisasi box & teks
            color = COLOR_MAP.get(cls_name, (0, 255, 0))
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            label_text = f"{cls_name.upper()} {score_val * 100:.1f}%"
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, y1), color, -1)
            cv2.putText(annotated, label_text, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Klasifikasi defect_type secara keseluruhan
        if found_cracked:
            defect_type = "cracked"
            has_defect = True
        elif found_dirt:
            defect_type = "dirt"
            has_defect = True
        else:
            defect_type = "good"
            has_defect = False

        # Status text di pojok kiri atas
        status_color = (0, 200, 0) if not has_defect else ((0, 0, 220) if defect_type == "cracked" else (0, 165, 255))
        status_msg = f"STATUS: {defect_type.upper()}"
        cv2.putText(annotated, status_msg, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2, cv2.LINE_AA)

        return {
            "has_defect": has_defect,
            "defect_type": defect_type,
            "detections": detections,
            "annotated_image": annotated
        }


def parse_args():
    parser = argparse.ArgumentParser(description="Inferensi Model RetinaNet")
    parser.add_argument(
        "--source",
        type=str,
        default="Egg Dataset Clean/test/images/cracked",
        help="Path ke file citra tunggal atau direktori citra"
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="runs/retinanet/best_retinanet.pth",
        help="Path ke file checkpoint model .pth"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="runs/retinanet/predictions",
        help="Folder penyimpanan citra hasil prediksi"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.40,
        help="Confidence threshold deteksi (default: 0.40)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Perangkat inferensi"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print("🔮 INFERENSI MODEL RETINANET (RESNET-50 FPN)")
    print("=" * 70)
    print(f"[*] Sumber    : {args.source}")
    print(f"[*] Bobot     : {args.weights}")
    print(f"[*] Threshold : {args.conf}")
    print("=" * 70)

    detector = RetinaNetDetector(
        weights_path=args.weights,
        device=args.device,
        conf_thresh=args.conf
    )

    os.makedirs(args.output_dir, exist_ok=True)

    # Identifikasi apakah input berupa file tunggal atau folder
    if os.path.isfile(args.source):
        img_paths = [args.source]
    else:
        valid_exts = ("*.jpg", "*.jpeg", "*.png")
        img_paths = []
        for ext in valid_exts:
            img_paths.extend(glob.glob(os.path.join(args.source, "**", ext), recursive=True))

    print(f"[*] Ditemukan {len(img_paths)} citra untuk diproses.\n")

    for idx, path in enumerate(img_paths, 1):
        result = detector.detect(path)
        base_name = os.path.basename(path)
        out_path = os.path.join(args.output_dir, f"pred_{base_name}")

        cv2.imwrite(out_path, result["annotated_image"])
        print(f"[{idx:03d}/{len(img_paths):03d}] {base_name} -> {result['defect_type'].upper()} ({len(result['detections'])} box) | Tersimpan di {out_path}")

    print("\n" + "=" * 70)
    print(f"✅ Seluruh inferensi selesai! Cek folder '{args.output_dir}'")
    print("=" * 70)


if __name__ == "__main__":
    main()

