import sys, os
sys.path.insert(0, r'C:\laragon\www\Egg-SortingSystem\Egg Sorting System')
import numpy as np
import cv2
from PIL import Image
from modules.shape_size_detector import preprocess_egg_image

img_path = r'C:/Users/THOR/.gemini/antigravity/brain/74a4bd7b-f41c-46ec-8fd3-481fe8d98149/.user_uploaded/media_1795099443896_525519d8.jpg'
pil = Image.open(img_path).convert('RGB')
arr_orig = np.array(pil)
h_orig, w_orig = arr_orig.shape[:2]
print(f"Original image size: {w_orig}x{h_orig}")

# Cek hasil deteksi elips pada dimensi asli
det_size = 640
scale_det = det_size / max(h_orig, w_orig)
arr_det = cv2.resize(arr_orig, (int(w_orig * scale_det), int(h_orig * scale_det)), interpolation=cv2.INTER_AREA)
h_det, w_det = arr_det.shape[:2]

hsv_det = cv2.cvtColor(arr_det, cv2.COLOR_RGB2HSV)
s = cv2.GaussianBlur(hsv_det[:, :, 1], (9, 9), 0)
_, mask_s = cv2.threshold(s, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
if mask_s[0, 0] > 128: mask_s = cv2.bitwise_not(mask_s)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
m_clean = cv2.morphologyEx(mask_s, cv2.MORPH_OPEN, kernel)
m_clean = cv2.morphologyEx(m_clean, cv2.MORPH_CLOSE, kernel)
cnts, _ = cv2.findContours(m_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
best_cnt = max(cnts, key=cv2.contourArea)
ellipse = cv2.fitEllipse(best_cnt)
(cx_d, cy_d), (ax1_d, ax2_d), angle = ellipse
cx_orig = cx_d / scale_det
cy_orig = cy_d / scale_det
ax1_orig = ax1_d / scale_det
ax2_orig = ax2_d / scale_det

bx1 = round(max(0.0, cx_orig - ax1_orig / 2.0), 2)
by1 = round(max(0.0, cy_orig - ax2_orig / 2.0), 2)
bx2 = round(min(float(w_orig), cx_orig + ax1_orig / 2.0), 2)
by2 = round(min(float(h_orig), cy_orig + ax2_orig / 2.0), 2)

print(f"Ellipse center: ({cx_orig:.1f}, {cy_orig:.1f}), axes: ({ax1_orig:.1f}, {ax2_orig:.1f})")
print(f"Calculated Bounding Box in original image (w={w_orig}, h={h_orig}):")
print(f"  x1={bx1}, y1={by1}, x2={bx2}, y2={by2}")
print(f"  Box width={bx2-bx1:.1f}, height={by2-by1:.1f}")

# Gambar box di atas citra asli dan simpan
vis = arr_orig.copy()
cv2.rectangle(vis, (int(bx1), int(by1)), (int(bx2), int(by2)), (0, 255, 0), 3)
cv2.imwrite(r'C:\laragon\www\Egg-SortingSystem\Egg Sorting System\bbox_test.jpg', cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
print("Saved visualization to bbox_test.jpg!")
