import sys, os
os.environ["YOLO_CONFIG_DIR"] = r'C:\laragon\www\Egg-SortingSystem\Egg Sorting System'
sys.path.insert(0, r'C:\laragon\www\Egg-SortingSystem\Egg Sorting System')
import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO
from modules.shape_size_detector import preprocess_egg_image, measure_egg_shape_and_size, get_clean_egg_canvas, detect_physical_cracks

model = YOLO(r'C:\laragon\www\Egg-SortingSystem\Egg Sorting System\models\best.pt')
CLASS_NAMES = ['cracked', 'dirt']

img_path = r'C:/Users/THOR/.gemini/antigravity/brain/74a4bd7b-f41c-46ec-8fd3-481fe8d98149/.user_uploaded/media_1795099443896_525519d8.jpg'
pil = Image.open(img_path).convert('RGB')
np_img = np.array(pil)

print("1. Running preprocess_egg_image...")
np_img = preprocess_egg_image(np_img, target_canvas=640, target_egg_height=416)
print("   Done, shape:", np_img.shape)

print("2. Running measure_egg_shape_and_size...")
shape_res = measure_egg_shape_and_size(np_img, base_px_to_mm=0.120)
print("   is_valid_egg:", shape_res.get('is_valid_egg') if shape_res else False)

ellipse_orig = shape_res.get("ellipse_orig")
np_normalized, scale, x_off, y_off = get_clean_egg_canvas(np_img, ellipse_orig)

print("3. Predicting with YOLO...")
preds_norm = model.predict(Image.fromarray(np_normalized), verbose=False, conf=0.10)[0]
boxes_norm = getattr(preds_norm, "boxes", None)
print("   preds_norm boxes:", len(boxes_norm) if boxes_norm is not None else 0)

preds_raw = model.predict(Image.fromarray(np_img), verbose=False, conf=0.10)[0]
boxes_raw = getattr(preds_raw, "boxes", None)
print("   preds_raw boxes:", len(boxes_raw) if boxes_raw is not None else 0)

print("4. Physical cracks...")
has_physical_crack, crack_details = detect_physical_cracks(np_img, ellipse_orig)
print("   has_physical_crack:", has_physical_crack)

# Final decision simulation
detections = []
if boxes_norm and len(boxes_norm) > 0:
    for box in boxes_norm:
        cls_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        detections.append({"class_name": CLASS_NAMES[cls_id], "confidence": conf})

if boxes_raw and len(boxes_raw) > 0:
    for box in boxes_raw:
        cls_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        detections.append({"class_name": CLASS_NAMES[cls_id], "confidence": conf})

cracked_boxes = [d for d in detections if d["class_name"] == "cracked" and d["confidence"] >= 0.25]
dirt_boxes = [d for d in detections if d["class_name"] == "dirt" and d["confidence"] >= 0.25]

if cracked_boxes:
    decision = "cracked"
elif dirt_boxes:
    decision = "dirt"
else:
    decision = "good"

print("\n>>> FINAL DECISION:", decision.upper(), "<<<")
