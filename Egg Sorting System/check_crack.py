import sys, os
os.environ["YOLO_CONFIG_DIR"] = r'C:\laragon\www\Egg-SortingSystem\Egg Sorting System'
sys.path.insert(0, r'C:\laragon\www\Egg-SortingSystem\Egg Sorting System')
import numpy as np
from PIL import Image
from ultralytics import YOLO
from modules.shape_size_detector import preprocess_egg_image, measure_egg_shape_and_size, get_clean_egg_canvas, detect_physical_cracks

model_path = r'C:\laragon\www\Egg-SortingSystem\Egg Sorting System\models\best.pt'
model = YOLO(model_path)
print("Model classes:", model.names)

img_path = r'C:/Users/THOR/.gemini/antigravity/brain/74a4bd7b-f41c-46ec-8fd3-481fe8d98149/.user_uploaded/media_1795099443896_525519d8.jpg'
pil = Image.open(img_path).convert('RGB')
np_img = np.array(pil)

# Preprocessing
np_pre = preprocess_egg_image(np_img, target_canvas=640, target_egg_height=416)
shape_res = measure_egg_shape_and_size(np_pre, base_px_to_mm=0.120)
ellipse_orig = shape_res.get("ellipse_orig")
np_norm, scale, x_off, y_off = get_clean_egg_canvas(np_pre, ellipse_orig)

print("\n--- 1. PREDICT PADA CANVASS NORMALIZED ---")
res_norm = model.predict(Image.fromarray(np_norm), conf=0.10, verbose=False)[0]
if res_norm.boxes:
    for b in res_norm.boxes:
        cls_id = int(b.cls[0].item())
        conf = float(b.conf[0].item())
        xyxy = b.xyxy[0].tolist()
        print(f"  NORM: cls={cls_id} ({model.names[cls_id]}), conf={conf:.4f}, xyxy={xyxy}")
else:
    print("  NORM: No boxes")

print("\n--- 2. PREDICT PADA PREPROCESSED IMAGE (np_pre) ---")
res_pre = model.predict(Image.fromarray(np_pre), conf=0.10, verbose=False)[0]
if res_pre.boxes:
    for b in res_pre.boxes:
        cls_id = int(b.cls[0].item())
        conf = float(b.conf[0].item())
        xyxy = b.xyxy[0].tolist()
        print(f"  PRE: cls={cls_id} ({model.names[cls_id]}), conf={conf:.4f}, xyxy={xyxy}")
else:
    print("  PRE: No boxes")

print("\n--- 3. OPENCV DETECT PHYSICAL CRACKS ---")
has_crack, crack_details = detect_physical_cracks(np_pre, ellipse_orig)
print(f"  OpenCV crack: has_crack={has_crack}, details={crack_details}")
