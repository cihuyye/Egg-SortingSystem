import sys, os
sys.path.insert(0, r'c:\laragon\www\Egg-SortingSystem')
import numpy as np
import cv2
from PIL import Image
from modules.shape_size_detector import preprocess_egg_image, measure_egg_shape_and_size

print("=" * 60)
print("TEST 1: Foto dari kamera HP (latar lantai keramik)")
print("=" * 60)
img_path = r'C:/Users/THOR/.gemini/antigravity/brain/74a4bd7b-f41c-46ec-8fd3-481fe8d98149/.user_uploaded/media_1795099443896_525519d8.jpg'
pil = Image.open(img_path).convert('RGB')
np_img = np.array(pil)
np_pre = preprocess_egg_image(np_img, target_canvas=640, target_egg_height=416)
res = measure_egg_shape_and_size(np_pre, base_px_to_mm=0.120)
print(f"  is_valid_egg : {res.get('is_valid_egg') if res else None}")
print(f"  size_category: {res.get('size_category') if res else 'N/A'}")
print(f"  shape_index  : {res.get('esi_percentage') if res else 'N/A'}%")
print(f"  weight_est   : {res.get('estimated_weight_gram') if res else 'N/A'}g")
print()

print("=" * 60)
print("TEST 2: Simulasi frame ESP32-CAM (320x240 dark background)")
print("=" * 60)
# Buat citra simulasi: telur di tengah background gelap (seperti konveyor)
fake_esp = np.zeros((240, 320, 3), dtype=np.uint8)
# Gambar ellipse (simulasi telur cokelat) di tengah
cy, cx = 120, 160
cv2.ellipse(fake_esp, (cx, cy), (60, 45), 15, 0, 360, (180, 140, 110), -1)
# Tambah noise sedikit
noise = np.random.randint(0, 15, fake_esp.shape, dtype=np.uint8)
fake_esp = np.clip(fake_esp.astype(int) + noise, 0, 255).astype(np.uint8)

np_pre2 = preprocess_egg_image(fake_esp, target_canvas=640, target_egg_height=416)
res2 = measure_egg_shape_and_size(np_pre2, base_px_to_mm=0.120)
print(f"  is_valid_egg : {res2.get('is_valid_egg') if res2 else None}")
print(f"  size_category: {res2.get('size_category') if res2 else 'N/A'}")
print()

print("=" * 60)
print("TEST 3: Non-egg (foto random) → harus BUKAN TELUR")
print("=" * 60)
fake_bukan = np.full((300, 300, 3), 200, dtype=np.uint8)  # kotak abu-abu
np_pre3 = preprocess_egg_image(fake_bukan, target_canvas=640, target_egg_height=416)
res3 = measure_egg_shape_and_size(np_pre3, base_px_to_mm=0.120)
print(f"  is_valid_egg : {res3.get('is_valid_egg') if res3 else None} (None = BUKAN TELUR ✓)")
print()
print("Semua test selesai.")
