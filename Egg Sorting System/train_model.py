import os
import torch
from ultralytics import YOLO

def train():
    device = '0' if torch.cuda.is_available() else 'cpu'
    print(f'[INFO] Menggunakan perangkat akselerasi: {device}')
    if device == '0':
        print(f'[INFO] Nama GPU: {torch.cuda.get_device_name(0)}')
        
    dataset_yaml = r'D:/Egg Dataset/Egg Dataset/data.yaml'
    
    if not os.path.exists(dataset_yaml):
        raise FileNotFoundError(f'File konfigurasi dataset tidak ditemukan di: {dataset_yaml}')
        
    print(f'[INFO] Memuat model pre-trained YOLOv8 Nano (yolov8n.pt)...')
    model = YOLO('yolov8n.pt')
    
    print('[INFO] Memulai proses pelatihan model YOLOv8...')
    results = model.train(
        data=dataset_yaml,
        epochs=50,
        imgsz=640,
        batch=16,
        device=device,
        project=r'D:/Egg Sorting System/runs',
        name='egg_defect_yolov8',
        exist_ok=True,
        verbose=True
    )
    
    best_weight = os.path.join(r'D:/Egg Sorting System/runs', 'egg_defect_yolov8', 'weights', 'best.pt')
    target_weight = r'D:/Egg Sorting System/models/best.pt'
    
    if os.path.exists(best_weight):
        import shutil
        shutil.copy2(best_weight, target_weight)
        print(f'\n[SUKSES] Bobot model terbaik berhasil disimpan di: {target_weight}')
    else:
        print(f'\n[SELESAI] Pelatihan selesai. Cek folder runs untuk hasil.')

if __name__ == '__main__':
    train()
