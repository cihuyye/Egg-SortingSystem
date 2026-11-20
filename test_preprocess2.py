import sys, os
sys.path.insert(0, r'c:\laragon\www\Egg-SortingSystem')
import numpy as np
import cv2
from PIL import Image
from modules.shape_size_detector import preprocess_egg_image, measure_egg_shape_and_size

img_path = r'C:/Users/THOR/.gemini/antigravity/brain/74a4bd7b-f41c-46ec-8fd3-481fe8d98149/.user_uploaded/media_1795099443896_525519d8.jpg'
pil = Image.open(img_path).convert('RGB')
np_img_orig = np.array(pil)
print("Citra asli:", np_img_orig.shape)

# --- Debug measure_egg_shape_and_size SEBELUM preprocessing ---
res_before = measure_egg_shape_and_size(np_img_orig)
print("SEBELUM - full result:", res_before)

# --- Preprocessing ---
np_preprocessed = preprocess_egg_image(np_img_orig, target_canvas=640, target_egg_height=416)
print("Setelah preprocessing:", np_preprocessed.shape)
# Cek apakah ada piksel non-hitam (ada objek)
nonblack = np.count_nonzero(np_preprocessed.sum(axis=2))
print(f"Jumlah piksel non-hitam di preprocessed: {nonblack} / {640*640}")

# Simpan preprocessed image
out_path = os.path.join(r'c:\laragon\www\Egg-SortingSystem', 'egg_preprocess_debug.jpg')
cv2.imwrite(out_path, cv2.cvtColor(np_preprocessed, cv2.COLOR_RGB2BGR))
print(f"Saved: {out_path}")

# --- Debug measure_egg_shape_and_size SETELAH preprocessing ---
res_after = measure_egg_shape_and_size(np_preprocessed)
print("SETELAH - full result:", res_after)
