import sys, os
sys.path.insert(0, r'c:\laragon\www\Egg-SortingSystem')
import numpy as np
import cv2
from PIL import Image
from modules.shape_size_detector import preprocess_egg_image

img_path = r'C:/Users/THOR/.gemini/antigravity/brain/74a4bd7b-f41c-46ec-8fd3-481fe8d98149/.user_uploaded/media_1795099443896_525519d8.jpg'
pil = Image.open(img_path).convert('RGB')
np_img_orig = np.array(pil)

# Preprocessing
np_preprocessed = preprocess_egg_image(np_img_orig, target_canvas=640, target_egg_height=416)

# Manual trace measure_egg_shape_and_size dengan logging
h_orig, w_orig = np_preprocessed.shape[:2]
print(f"Input ke measure: {h_orig}x{w_orig}")

standard_dim = 640.0
scale_factor = standard_dim / max(h_orig, w_orig)
new_w = max(10, int(w_orig * scale_factor))
new_h = max(10, int(h_orig * scale_factor))
print(f"scale_factor={scale_factor}, new_w={new_w}, new_h={new_h}")
img_scaled = cv2.resize(np_preprocessed, (new_w, new_h), interpolation=cv2.INTER_AREA)

gray = cv2.cvtColor(img_scaled, cv2.COLOR_RGB2GRAY)
blurred = cv2.GaussianBlur(gray, (7, 7), 0)
_, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

corners = [float(gray[0, 0]), float(gray[0, -1]), float(gray[-1, 0]), float(gray[-1, -1])]
is_bright_bg = np.mean(corners) > 100
print(f"Corner values: {corners}, is_bright_bg={is_bright_bg}")

if is_bright_bg:
    thresh = cv2.bitwise_not(thresh)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"Jumlah kontur ditemukan: {len(contours)}")

scaled_total_area = new_h * new_w
print(f"total_area={scaled_total_area}")

for i, cnt in enumerate(contours[:5]):
    area = cv2.contourArea(cnt)
    area_ratio = area / scaled_total_area
    print(f"  Kontur {i}: area={area:.0f}, ratio={area_ratio:.3f} (harus 0.025-0.85)")
    if len(cnt) >= 5 and 0.025 <= area_ratio <= 0.85:
        try:
            ellipse = cv2.fitEllipse(cnt)
            (cx, cy), (ax1, ax2), angle = ellipse
            minor = min(ax1, ax2)
            major = max(ax1, ax2)
            aspect = minor / major
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            print(f"    ellipse: cx={cx:.0f},cy={cy:.0f}, ax1={ax1:.0f},ax2={ax2:.0f}")
            print(f"    aspect={aspect:.3f} (harus 0.55-0.95), solidity={solidity:.3f} (harus >=0.85)")
            
            # Cek warna
            mask_s = np.zeros((new_h, new_w), dtype=np.uint8)
            cv2.drawContours(mask_s, [cnt], -1, 255, -1)
            mean_rgb = cv2.mean(img_scaled, mask=mask_s)[:3]
            r, g, b = mean_rgb
            is_warm_brown = (r > b * 1.04 and r > g * 0.88 and r > 60)
            is_white_shell = (r > 120 and g > 120 and b > 110 and abs(r - b) < 45)
            print(f"    mean_rgb=({r:.0f},{g:.0f},{b:.0f}), warm_brown={is_warm_brown}, white_shell={is_white_shell}")
            
            # Cek ukuran mm
            length_mm = major * 0.135
            width_mm = minor * 0.135
            print(f"    length_mm={length_mm:.1f} (harus 36-78), width_mm={width_mm:.1f} (harus 26-58)")
        except Exception as e:
            print(f"    ERROR: {e}")
    
print("Done debug")
