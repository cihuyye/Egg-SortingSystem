import sys
sys.path.insert(0, r'C:\laragon\www\Egg-SortingSystem\Egg Sorting System')
import numpy as np
import cv2
from PIL import Image
from modules.shape_size_detector import preprocess_egg_image, validate_egg_candidate

img_path = r'C:/Users/THOR/.gemini/antigravity/brain/74a4bd7b-f41c-46ec-8fd3-481fe8d98149/.user_uploaded/media_1795099443896_525519d8.jpg'
pil = Image.open(img_path).convert('RGB')
np_img = np.array(pil)
print("Citra asli shape:", np_img.shape)

np_pre = preprocess_egg_image(np_img, target_canvas=640, target_egg_height=416)
print("np_pre shape:", np_pre.shape)

# Sekarang kita telusuri persis apa yang dilakukan measure_egg_shape_and_size
image_np = np_pre
base_px_to_mm = 0.120

h_orig, w_orig = image_np.shape[:2]
standard_dim = 640.0
scale_factor = standard_dim / max(h_orig, w_orig)
new_w = max(10, int(w_orig * scale_factor))
new_h = max(10, int(h_orig * scale_factor))
img_scaled = cv2.resize(image_np, (new_w, new_h), interpolation=cv2.INTER_AREA)

gray = cv2.cvtColor(img_scaled, cv2.COLOR_RGB2GRAY)
blurred = cv2.GaussianBlur(gray, (7, 7), 0)

_, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

corners = [float(gray[0, 0]), float(gray[0, -1]), float(gray[-1, 0]), float(gray[-1, -1])]
is_bright_bg = np.mean(corners) > 100
print(f"corners: {corners}, is_bright_bg: {is_bright_bg}")
if is_bright_bg:
    thresh = cv2.bitwise_not(thresh)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"Total contours: {len(contours)}")

valid_candidates = []
scaled_total_area = new_h * new_w
for i, cnt in enumerate(contours):
    area = cv2.contourArea(cnt)
    print(f"cnt {i} area: {area}, ratio: {area/scaled_total_area}")
    ok, ellipse, score = validate_egg_candidate(img_scaled, cnt, scaled_total_area)
    print(f"  validate_egg_candidate: ok={ok}, score={score}")
    if ok and ellipse is not None:
        valid_candidates.append((cnt, ellipse, score))

if not valid_candidates:
    print("FAILED: no valid candidates!")
    sys.exit(0)

best_cnt, best_ellipse_scaled, _ = max(valid_candidates, key=lambda c: c[2])
(cx_s, cy_s), (ax1_s, ax2_s), angle = best_ellipse_scaled

minor_axis = min(ax1_s, ax2_s)
major_axis = max(ax1_s, ax2_s)

length_mm = major_axis * base_px_to_mm
width_mm = minor_axis * base_px_to_mm
print(f"length_mm: {length_mm}, width_mm: {width_mm}")

if not (36.0 <= length_mm <= 78.0) or not (26.0 <= width_mm <= 58.0):
    print(f"FAILED length/width bounds: length={length_mm}, width={width_mm}")
    sys.exit(0)

si = (width_mm / length_mm) * 100.0 if length_mm > 0 else 0
print(f"SI: {si}")
if not (58.0 <= si <= 94.0):
    print(f"FAILED SI bounds: si={si}")
    sys.exit(0)

l_cm = length_mm / 10.0
w_cm = width_mm / 10.0
vol_cm3 = (np.pi / 6.0) * l_cm * (w_cm ** 2)
weight_gram = vol_cm3 * 1.035
print(f"weight_gram: {weight_gram}")

if not (26.0 <= weight_gram <= 88.0):
    print(f"FAILED weight_gram bounds: weight_gram={weight_gram}")
    sys.exit(0)

print("SUCCESS! All conditions passed!")
