import sys, os
os.environ["YOLO_CONFIG_DIR"] = r'C:\laragon\www\Egg-SortingSystem\Egg Sorting System'
sys.path.insert(0, r'C:\laragon\www\Egg-SortingSystem\Egg Sorting System')
import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO

model = YOLO(r'C:\laragon\www\Egg-SortingSystem\Egg Sorting System\models\best.pt')

# Kita jalankan EXACT kode dari notebook preprocess_dataset.ipynb
def notebook_preprocess(arr_orig, target_size=(640, 640), target_egg_height=416.0, ellipse_scale=0.975, blur_kernel=(7, 7)):
    h_orig, w_orig = arr_orig.shape[:2]
    det_size = 640
    scale_det = det_size / max(h_orig, w_orig)
    arr_det = cv2.resize(arr_orig, (int(w_orig * scale_det), int(h_orig * scale_det)), interpolation=cv2.INTER_AREA)
    h_det, w_det = arr_det.shape[:2]

    hsv_det = cv2.cvtColor(arr_det, cv2.COLOR_RGB2HSV)
    corners = np.vstack([arr_det[:10, :10], arr_det[:10, -10:], arr_det[-10:, :10], arr_det[-10:, -10:]])
    bg_median = np.median(corners, axis=(0, 1))

    s = cv2.GaussianBlur(hsv_det[:, :, 1], (9, 9), 0)
    _, mask_s = cv2.threshold(s, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if mask_s[0, 0] > 128: mask_s = cv2.bitwise_not(mask_s)

    color_diff = np.sqrt(np.sum((arr_det.astype(np.float32) - bg_median) ** 2, axis=2))
    diff_norm = np.clip(color_diff / (color_diff.max() + 1e-6) * 255, 0, 255).astype(np.uint8)
    diff_blur = cv2.GaussianBlur(diff_norm, (9, 9), 0)
    _, mask_diff = cv2.threshold(diff_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if mask_diff[0, 0] > 128: mask_diff = cv2.bitwise_not(mask_diff)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    best_ellipse = None
    best_score = -1.0

    for raw_mask in [mask_s, mask_diff]:
        m_clean = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, kernel)
        m_clean = cv2.morphologyEx(m_clean, cv2.MORPH_CLOSE, kernel)
        cnts, _ = cv2.findContours(m_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for cnt in cnts:
            area = cv2.contourArea(cnt)
            if area < (0.01 * h_det * w_det): continue
            if len(cnt) < 5: continue
            try: ellipse = cv2.fitEllipse(cnt)
            except: continue
            (cx, cy), (ax1, ax2), angle = ellipse
            minor, major = min(ax1, ax2), max(ax1, ax2)
            if minor <= 0 or major <= 0: continue
            aspect = minor / major
            if not (0.45 <= aspect <= 1.0): continue
            hull = cv2.convexHull(cnt)
            solidity = area / max(cv2.contourArea(hull), 1)
            if solidity < 0.75: continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            touches = (x <= 2 or y <= 2 or (x + bw) >= w_det - 2 or (y + bh) >= h_det - 2)
            if touches and area < 0.5 * h_det * w_det: continue
            mask_single = np.zeros((h_det, w_det), dtype=np.uint8)
            cv2.drawContours(mask_single, [cnt], -1, 255, -1)
            cnt_sat = cv2.mean(hsv_det[:, :, 1], mask=mask_single)[0]
            score = solidity * 100.0 + (1.0 - abs(aspect - 0.75)) * 50.0 + (cnt_sat / 255.0) * 50.0
            if score > best_score:
                best_score = score
                best_ellipse = ellipse

    if best_ellipse is None:
        return cv2.resize(arr_orig, target_size)

    (cx_d, cy_d), (ax1_d, ax2_d), angle = best_ellipse
    cx_orig = cx_d / scale_det
    cy_orig = cy_d / scale_det
    ax1_orig = ax1_d / scale_det
    ax2_orig = ax2_d / scale_det
    major_orig = max(ax1_orig, ax2_orig)

    norm_scale = target_egg_height / major_orig
    target_cx = target_size[0] / 2.0
    target_cy = target_size[1] / 2.0
    tx = target_cx - cx_orig * norm_scale
    ty = target_cy - cy_orig * norm_scale
    M = np.array([[norm_scale, 0, tx], [0, norm_scale, ty]], dtype=np.float32)
    s4_centered = cv2.warpAffine(arr_orig, M, target_size, flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    new_ax1 = ax1_orig * norm_scale * ellipse_scale
    new_ax2 = ax2_orig * norm_scale * ellipse_scale
    mask_2x = np.zeros((target_size[1] * 2, target_size[0] * 2), dtype=np.uint8)
    ellipse_2x = ((target_cx * 2, target_cy * 2), (new_ax1 * 2, new_ax2 * 2), angle)
    cv2.ellipse(mask_2x, ellipse_2x, 255, -1)
    mask_smooth = cv2.resize(mask_2x, target_size, interpolation=cv2.INTER_AREA)
    s5_mask = cv2.GaussianBlur(mask_smooth, blur_kernel, 0)

    mask_3ch = np.repeat((s5_mask / 255.0)[:, :, np.newaxis], 3, axis=2)
    s6_final = (s4_centered * mask_3ch).astype(np.uint8)
    return s6_final

img_path = r'C:/Users/THOR/.gemini/antigravity/brain/74a4bd7b-f41c-46ec-8fd3-481fe8d98149/.user_uploaded/media_1795099443896_525519d8.jpg'
pil = Image.open(img_path).convert('RGB')
arr_orig = np.array(pil)

s6 = notebook_preprocess(arr_orig)
cv2.imwrite(r'C:\laragon\www\Egg-SortingSystem\Egg Sorting System\s6_test.jpg', cv2.cvtColor(s6, cv2.COLOR_RGB2BGR))
print("s6 shape:", s6.shape)

res = model.predict(Image.fromarray(s6), conf=0.10, verbose=False)[0]
print("Boxes at conf=0.10:", len(res.boxes) if res.boxes else 0)
if res.boxes:
    for b in res.boxes:
        cls_id = int(b.cls[0].item())
        conf = float(b.conf[0].item())
        print(f"  cls={cls_id} ({model.names[cls_id]}), conf={conf:.4f}")

res25 = model.predict(Image.fromarray(s6), conf=0.25, verbose=False)[0]
print("Boxes at conf=0.25:", len(res25.boxes) if res25.boxes else 0)
if res25.boxes:
    for b in res25.boxes:
        cls_id = int(b.cls[0].item())
        conf = float(b.conf[0].item())
        print(f"  conf=0.25: cls={cls_id} ({model.names[cls_id]}), conf={conf:.4f}")
