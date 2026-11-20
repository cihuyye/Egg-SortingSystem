import sys
sys.path.insert(0, r'C:\laragon\www\Egg-SortingSystem\Egg Sorting System')
import numpy as np
import cv2
from PIL import Image
from modules.shape_size_detector import preprocess_egg_image, measure_egg_shape_and_size

print("=" * 60)
print("TEST 1: Foto HP Telur di Lantai Keramik")
print("=" * 60)
img_path = r'C:/Users/THOR/.gemini/antigravity/brain/74a4bd7b-f41c-46ec-8fd3-481fe8d98149/.user_uploaded/media_1795099443896_525519d8.jpg'
pil = Image.open(img_path).convert('RGB')
np_img = np.array(pil)
np_pre = preprocess_egg_image(np_img, target_canvas=640, target_egg_height=416)
res = measure_egg_shape_and_size(np_pre, base_px_to_mm=0.120)
print("Test 1 Result:", res)
assert res is not None and res.get("is_valid_egg") == True, "Test 1 GAGAL!"
print(">>> TEST 1: BERHASIL! <<<\n")

print("=" * 60)
print("TEST 2: ESP32-CAM Simulasi LED Putih / Top-Light Cool White")
print("=" * 60)
# Buat citra simulasi ESP32-CAM: telur cokelat disinari lampu putih dingin (B agak tinggi)
canvas_sim = np.zeros((240, 320, 3), dtype=np.uint8)
cv2.ellipse(canvas_sim, (160, 120), (55, 42), 10, 0, 360, (140, 125, 120), -1) # R=140, G=125, B=120
np_pre2 = preprocess_egg_image(canvas_sim, target_canvas=640, target_egg_height=416)
res2 = measure_egg_shape_and_size(np_pre2, base_px_to_mm=0.120)
print("Test 2 Result:", res2)
assert res2 is not None and res2.get("is_valid_egg") == True, "Test 2 GAGAL!"
print(">>> TEST 2: BERHASIL! <<<\n")

print("=" * 60)
print("TEST 3: Non-Egg Objek (Kotak Abu-Abu / Screenshot)")
print("=" * 60)
fake_box = np.full((300, 300, 3), 180, dtype=np.uint8)
np_pre3 = preprocess_egg_image(fake_box, target_canvas=640, target_egg_height=416)
res3 = measure_egg_shape_and_size(np_pre3, base_px_to_mm=0.120)
print("Test 3 Result (harus None):", res3)
assert res3 is None, "Test 3 GAGAL (seharusnya None)!"
print(">>> TEST 3: BERHASIL (Bukan Telur Ditolak)! <<<\n")

print("SEMUA TEST LULUS 100%!")
