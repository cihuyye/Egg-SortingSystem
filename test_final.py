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

# Preprocessing
np_preprocessed = preprocess_egg_image(np_img_orig, target_canvas=640, target_egg_height=416)
print("Preprocessed:", np_preprocessed.shape)

# Test dengan px_to_mm=0.120
res = measure_egg_shape_and_size(np_preprocessed, base_px_to_mm=0.120)
print("Result (px_to_mm=0.120):", res)
if res:
    print("  is_valid_egg:", res.get("is_valid_egg"))
    print("  length_mm:", res.get("length_mm"))
    print("  width_mm:", res.get("width_mm"))
    print("  shape_index:", res.get("esi_percentage"))
    print("  size_category:", res.get("size_category"))
    print("  estimated_weight:", res.get("estimated_weight_gram"))
