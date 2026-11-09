import os
import json

# 1. Definisikan ID kelas sesuai data.yaml
# Pastikan nama ini sama persis dengan yang kamu ketik di Labelme
class_mapping = {
    "crack": 0,
    "dirt": 1
}

# 2. Path relatif ke folder dataset utama
dataset_path = "Egg Dataset Clean"

# 3. Tentukan folder split dan folder kelas yang punya file JSON
splits = ["train", "valid", "test"]
classes = ["cracked", "dirt"] # Folder good diabaikan karena tidak ada JSON/labelnya

def convert_labelme_to_yolo():
    for split in splits:
        for cls in classes:
            # Path sumber JSON (di dalam images) dan tujuan TXT (di dalam labels)
            img_dir = os.path.join(dataset_path, split, "images", cls)
            label_dir = os.path.join(dataset_path, split, "labels", cls)

            # Lewati jika folder tidak ditemukan
            if not os.path.exists(img_dir):
                continue

            # Buat folder labels/... jika belum ada
            os.makedirs(label_dir, exist_ok=True)

            # Cari semua file JSON di folder images
            for filename in os.listdir(img_dir):
                if filename.endswith(".json"):
                    json_path = os.path.join(img_dir, filename)

                    # Buka dan baca file JSON Labelme
                    with open(json_path, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    img_w = data["imageWidth"]
                    img_h = data["imageHeight"]

                    # Siapkan nama file .txt keluaran
                    txt_filename = filename.replace(".json", ".txt")
                    txt_path = os.path.join(label_dir, txt_filename)

                    # Tulis hasil konversi ke file .txt di folder labels
                    with open(txt_path, "w", encoding="utf-8") as out_file:
                        for shape in data["shapes"]:
                            label = shape["label"]
                            if label not in class_mapping:
                                print(f"[WARNING] Label '{label}' tidak dikenal pada file {filename}. Cek typo di Labelme!")
                                continue

                            class_id = class_mapping[label]
                            points = shape["points"]

                            # Normalisasi koordinat poligon menjadi skala 0.0 - 1.0
                            yolo_points = []
                            for x, y in points:
                                norm_x = max(0.0, min(1.0, x / img_w))
                                norm_y = max(0.0, min(1.0, y / img_h))
                                yolo_points.append(f"{norm_x:.6f} {norm_y:.6f}")

                            # Format YOLO Segmentation: [class_id] [x1] [y1] [x2] [y2] ...
                            line = f"{class_id} " + " ".join(yolo_points) + "\n"
                            out_file.write(line)
                            
                    print(f"[SUCCESS] {txt_filename} -> tersimpan di {label_dir}")

if __name__ == "__main__":
    print("Memulai konversi JSON Labelme ke YOLOv8 TXT...")
    convert_labelme_to_yolo()
    print("Selesai! Semua file berhasil dikonversi ke folder labels masing-masing.")