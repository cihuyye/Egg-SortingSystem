import os
import argparse
import shutil
import torch
from ultralytics import YOLO

def train(epochs: int = 50, batch: int = 16, imgsz: int = 640):
    device = '0' if torch.cuda.is_available() else 'cpu'
    print(f"[INFO] Menggunakan perangkat akselerasi: {device}")
    if device == '0':
        print(f"[INFO] Nama GPU: {torch.cuda.get_device_name(0)}")
        
    dataset_yaml = r"D:/github_rama/Egg-SortingSystem/Egg Dataset Clean/data.yaml"
    
    if not os.path.exists(dataset_yaml):
        raise FileNotFoundError(f"File konfigurasi dataset tidak ditemukan di: {dataset_yaml}")
        
    base_dir = os.path.dirname(os.path.abspath(__file__))
    local_yolo = os.path.join(base_dir, "yolov8n.pt")
    weights_init = local_yolo if os.path.exists(local_yolo) else "yolov8n.pt"
    
    print(f"[INFO] Memuat model pre-trained ({weights_init})...")
    model = YOLO(weights_init)
    
    print(f"[INFO] Memulai proses pelatihan model YOLOv8 ({epochs} epoch, batch {batch}, imgsz {imgsz})...")
    print(f"[INFO] Dataset: {dataset_yaml} (2 Defect Classes: cracked, dirt. Good = background)")
    
    project_runs = r"D:/Egg Sorting System/runs"
    results = model.train(
        data=dataset_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project_runs,
        name="egg_defect_yolov8",
        exist_ok=True,
        verbose=True
    )
    
    best_weight = os.path.join(project_runs, "egg_defect_yolov8", "weights", "best.pt")
    targets = [
        r"D:/Egg Sorting System/models/best.pt",
        r"D:/github_rama/Egg-SortingSystem/Egg Sorting System/models/best.pt"
    ]
    
    if os.path.exists(best_weight):
        for target_weight in targets:
            target_dir = os.path.dirname(target_weight)
            os.makedirs(target_dir, exist_ok=True)
            if os.path.exists(target_weight):
                backup_path = os.path.join(target_dir, "best_prev_backup.pt")
                try:
                    shutil.copy2(target_weight, backup_path)
                    print(f"[INFO] Backup bobot sebelumnya disimpan di: {backup_path}")
                except Exception:
                    pass
            shutil.copy2(best_weight, target_weight)
            print(f"[SUKSES] Bobot model terbaik berhasil disimpan di: {target_weight}")
    else:
        print(f"\n[SELESAI] Pelatihan selesai. Cek folder {project_runs} untuk file weights.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Latih Model YOLOv8 Egg Defect Detection")
    parser.add_argument("--epochs", type=int, default=50, help="Jumlah epoch pelatihan (default: 50)")
    parser.add_argument("--batch", type=int, default=16, help="Ukuran batch (default: 16)")
    parser.add_argument("--imgsz", type=int, default=640, help="Resolusi citra (default: 640)")
    args = parser.parse_args()

    train(epochs=args.epochs, batch=args.batch, imgsz=args.imgsz)
