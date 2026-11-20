import sys
sys.path.insert(0, r'C:\laragon\www\Egg-SortingSystem\Egg Sorting System')
import numpy as np
from PIL import Image
from modules.shape_size_detector import preprocess_egg_image, measure_egg_shape_and_size

img_path = r'C:/Users/THOR/.gemini/antigravity/brain/74a4bd7b-f41c-46ec-8fd3-481fe8d98149/.user_uploaded/media_1795099443896_525519d8.jpg'
pil = Image.open(img_path).convert('RGB')
np_img = np.array(pil)
np_pre = preprocess_egg_image(np_img, target_canvas=640, target_egg_height=416)
res = measure_egg_shape_and_size(np_pre, base_px_to_mm=0.120)
print("is_valid_egg:", res.get("is_valid_egg") if res else None)
print("size_category:", res.get("size_category") if res else "N/A")
print("shape_index:", res.get("esi_percentage") if res else "N/A")
print("LULUS!" if res and res.get("is_valid_egg") else "GAGAL")
