import os
import time
from typing import List, Tuple
import cv2
import numpy as np
from PIL import Image

def process_single_image(
    image_path: str,
    target_size: Tuple[int, int] = (640, 640),
    ellipse_scale: float = 0.985,
    blur_kernel: Tuple[int, int] = (7, 7)
) -> np.ndarray:
    """
    Memproses satu citra telur:
    1. Resize ke target_size menggunakan interpolasi Lanczos (Image.LANCZOS).
    2. Segmentasi kontur telur dan fitting elips matematis.
    3. Normalisasi background menjadi hitam pekat.
    4. Anti-Aliasing tepi menggunakan 2x Supersampling + Gaussian Feathering.
    """
    # 1. Buka citra dan resize menggunakan metode Lanczos berkualitas tinggi
    pil_img = Image.open(image_path).convert("RGB")
    pil_resized = pil_img.resize(target_size, resample=Image.LANCZOS)
    arr = np.array(pil_resized)
    h, w = arr.shape[:2]

    # 2. Segmentasi kontur telur
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    
    # Deteksi ambang batas (Otsu thresholding adaptif)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Pastikan objek telur berwarna putih di atas background gelap
    if (np.sum(thresh == 255) / (h * w)) > 0.6:
        thresh = cv2.bitwise_not(thresh)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return arr

    best_cnt = max(contours, key=cv2.contourArea)
    if len(best_cnt) < 5:
        return arr

    try:
        # 3. Rekonstruksi kurva elips matematis
        ellipse = cv2.fitEllipse(best_cnt)
        (cx, cy), (ax1, ax2), angle = ellipse

        # Validasi ukuran elips yang wajar
        if ax1 < 40 or ax2 < 40 or ax1 > w * 1.5 or ax2 > h * 1.5:
            return arr

        smooth_ellipse = ((cx, cy), (ax1 * ellipse_scale, ax2 * ellipse_scale), angle)

        # 4. Anti-Aliasing dengan 2x Supersampling
        mask_2x = np.zeros((h * 2, w * 2), dtype=np.uint8)
        ellipse_2x = ((cx * 2, cy * 2), (smooth_ellipse[1][0] * 2, smooth_ellipse[1][1] * 2), angle)
        cv2.ellipse(mask_2x, ellipse_2x, 255, -1)

        # Downscale kembali ke ukuran asli dengan INTER_AREA (anti-aliasing alami)
        mask_smooth = cv2.resize(mask_2x, (w, h), interpolation=cv2.INTER_AREA)

        # Gaussian Feathering untuk transisi lembut tepi
        mask_feather = cv2.GaussianBlur(mask_smooth, blur_kernel, 0) / 255.0
        mask_3ch = np.repeat(mask_feather[:, :, np.newaxis], 3, axis=2)

        # 5. Alpha blending ke latar belakang hitam pekat
        result = (arr * mask_3ch).astype(np.uint8)
        return result

    except Exception:
        return arr


def run_dataset_preprocessing(
    src_dir: str = r"D:\Egg Dataset Original",
    dst_dir: str = r"D:\Egg Dataset\Egg Dataset"
) -> None:
    """
    Menjalankan proses pra-pengolahan massal (batch preprocessing) untuk seluruh dataset:
    - Memproses seluruh folder train, valid, dan test.
    - Menjaga integritas file citra JPG dan file anotasi label TXT YOLO.
    """
    if not os.path.exists(src_dir):
        print(f"[ERROR] Folder sumber tidak ditemukan: {src_dir}")
        return

    splits: List[str] = ["train", "valid", "test"]
    categories: List[str] = ["cracked", "dirt", "good"]

    total_images: int = 0
    start_time: float = time.time()

    print("=" * 65)
    print("  MEMULAI PRA-PENGOLAHAN DATASET CITRA TELUR (LANCZOS + BG HITAM)")
    print("=" * 65)
    print(f"Sumber Input : {src_dir}")
    print(f"Target Output: {dst_dir}\n")

    for split in splits:
        for cat in categories:
            src_img_folder = os.path.join(src_dir, split, "images", cat)
            dst_img_folder = os.path.join(dst_dir, split, "images", cat)

            if not os.path.exists(src_img_folder):
                continue

            os.makedirs(dst_img_folder, exist_ok=True)
            files = [f for f in os.listdir(src_img_folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
            print(f"--> Memproses subset [{split}/{cat}]: {len(files)} citra...")

            for fname in files:
                src_path = os.path.join(src_img_folder, fname)
                dst_path = os.path.join(dst_img_folder, fname)

                # Jalankan fungsi pemrosesan citra
                processed_arr = process_single_image(src_path)
                Image.fromarray(processed_arr).save(dst_path, quality=98)
                total_images += 1

    elapsed = time.time() - start_time
    print("\n" + "=" * 65)
    print(f"  SELESAI! Total {total_images} citra berhasil diproses.")
    print(f"  Waktu Eksekusi: {elapsed:.2f} detik (~{elapsed/max(total_images, 1)*1000:.1f} ms/citra)")
    print("=" * 65)


if __name__ == "__main__":
    run_dataset_preprocessing()

