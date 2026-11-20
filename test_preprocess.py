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

res_before = measure_egg_shape_and_size(np_img_orig)
valid_before = res_before.get("is_valid_egg") if res_before else None
print("Shape detect SEBELUM preprocessing: is_valid_egg =", valid_before)

np_preprocessed = preprocess_egg_image(np_img_orig, target_canvas=640, target_egg_height=416)
print("Citra setelah preprocessing:", np_preprocessed.shape)

res_after = measure_egg_shape_and_size(np_preprocessed)
valid_after = res_after.get("is_valid_egg") if res_after else None
print("Shape detect SETELAH preprocessing: is_valid_egg =", valid_after)
if res_after:
    lmm = res_after.get("length_mm", 0)
    wmm = res_after.get("width_mm", 0)
    esi = res_after.get("shape_index_esi", 0)
    print(f"  length_mm={lmm:.1f}, width_mm={wmm:.1f}, shape_index={esi:.3f}")

os.makedirs('c:/temp', exist_ok=True)
cv2.imwrite('c:/temp/egg_preprocessed_test.jpg', cv2.cvtColor(np_preprocessed, cv2.COLOR_RGB2BGR))
print("Tersimpan ke c:/temp/egg_preprocessed_test.jpg")
