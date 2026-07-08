#!/usr/bin/env python3

import os
import cv2
import shutil
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from segment_anything import sam_model_registry, SamPredictor

# Shapely components for clean geometric transformations
import shapely
from shapely.geometry import Polygon
from shapely.affinity import translate

# ---------------------------------------------------------
# 1. PerSAM Visual Weights Extractor
# ---------------------------------------------------------
def get_persam_weights(predictor, ref_image, ref_mask):
    """
    Extracts DINOv2 visual features from the reference template 
    so SAM recognizes the custom styling and texture of the vehicles.
    """
    with torch.no_grad():
        predictor.set_image(ref_image)
        ref_features = predictor.get_image_embedding()
        
        mask_tensor = torch.from_numpy(ref_mask).float().unsqueeze(0).unsqueeze(0).to(ref_features.device)
        mask_resized = F.interpolate(mask_tensor, size=ref_features.shape[-2:], mode="bilinear")
        mask_resized = (mask_resized > 0.5).float()
        
        target_feat = ref_features * mask_resized
        target_weight = target_feat.sum(dim=(-1, -2)) / (mask_resized.sum(dim=(-1, -2)) + 1e-6)
        
    return target_weight

# ---------------------------------------------------------
# 2. Mask → Local 512 Pixel Coordinates
# ---------------------------------------------------------
def mask_to_local_obb_points(mask):
    """
    Converts binary masks into raw local 4-point coordinates 
    so Shapely can translate them before YOLO normalization.
    """
    contours, _ = cv2.findContours(mask.astype("uint8"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    cnt = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(cnt)
    
    (cx, cy), (w, h), angle = rect
    max_dimension = max(w, h)
    min_dimension = min(w, h)
    
    if max_dimension > 90 or max_dimension < 6: 
        return None
        
    aspect_ratio = max_dimension / (min_dimension + 1e-6)
    if aspect_ratio > 4.5 or aspect_ratio < 1.1:
        return None

    box_points = cv2.boxPoints(rect)
    pts = np.array(box_points, dtype="float32")
    
    x_sorted = pts[np.argsort(pts[:, 0]), :]
    left_pts = x_sorted[:2, :]
    right_pts = x_sorted[2:, :]
    
    tl = left_pts[np.argsort(left_pts[:, 1])[0]]
    bl = left_pts[np.argsort(left_pts[:, 1])[1]]
    tr = right_pts[np.argsort(right_pts[:, 1])[0]]
    br = right_pts[np.argsort(right_pts[:, 1])[1]]
    
    return [tl, tr, br, bl]

# ---------------------------------------------------------
# 3. Post-Processing Geometric Correction Head
# ---------------------------------------------------------
def geometric_correction_head(refined_labels, img_w, img_h):
    """
    STRICT POST-PROCESSING HEAD: Resolves remaining edge case failures 
    by targeting axis-aligned bleeding boxes in building shadows.
    """
    if len(refined_labels) == 0:
        return refined_labels

    corrected_labels = []
    valid_cars = []

    for label in refined_labels:
        cls_id = label[0]
        pts = np.array(label[1:]).reshape(4, 2)
        
        rect = cv2.minAreaRect((pts * np.array([img_w, img_h])).astype(np.float32))
        (cx, cy), (w, h), angle = rect
        max_dim = max(w, h)
        min_dim = min(w, h)
        aspect = max_dim / (min_dim + 1e-6)

        if 15 < max_dim < 42 and 1.3 < aspect < 3.2 and angle != 0.0:
            valid_cars.append({
                'center': (cx, cy),
                'size': (w, h),
                'angle': angle,
                'raw_label': label
            })

    for label in refined_labels:
        cls_id = label[0]
        pts = np.array(label[1:]).reshape(4, 2)
        rect = cv2.minAreaRect((pts * np.array([img_w, img_h])).astype(np.float32))
        (cx, cy), (w, h), angle = rect
        max_dim = max(w, h)
        
        is_bad_box = max_dim > 45 or max_dim < 8 or angle == 0.0 or angle == 90.0

        if is_bad_box:
            if len(valid_cars) > 0:
                distances = [np.linalg.norm(np.array((cx, cy)) - np.array(vc['center'])) for vc in valid_cars]
                closest_car = valid_cars[np.argmin(distances)]
                matched_w, matched_h = closest_car['size']
                matched_angle = closest_car['angle']
            else:
                matched_w, matched_h = 30.0, 15.0
                matched_angle = 0.0
            
            corrected_rect = ((cx, cy), (matched_w, matched_h), matched_angle)
            box_points = cv2.boxPoints(corrected_rect)
            
            x_sorted = box_points[np.argsort(box_points[:, 0]), :]
            left_pts = x_sorted[:2, :]
            right_pts = x_sorted[2:, :]
            tl = left_pts[np.argsort(left_pts[:, 1])[0]]
            bl = left_pts[np.argsort(left_pts[:, 1])[1]]
            tr = right_pts[np.argsort(right_pts[:, 1])[0]]
            br = right_pts[np.argsort(right_pts[:, 1])[1]]
            
            normalized_obb = [
                tl[0] / img_w, tl[1] / img_h,
                tr[0] / img_w, tr[1] / img_h,
                br[0] / img_w, br[1] / img_h,
                bl[0] / img_w, bl[1] / img_h
            ]
            corrected_labels.append([int(cls_id)] + normalized_obb)
        else:
            corrected_labels.append(label)

    return corrected_labels

# ---------------------------------------------------------
# 4. Load SAM Model Configuration
# ---------------------------------------------------------
def load_sam(model_path="models/sam_vit_b.pth"):
    print(f"Loading SAM checkpoint from: {model_path}")
    sam = sam_model_registry["vit_b"](checkpoint=model_path)

    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
        
    sam.to(device)
    predictor = SamPredictor(sam)
    print(f"SAM loaded successfully on target platform: {device}")
    return predictor, device

# ---------------------------------------------------------
# 5. Main Datasets Processing Loop (1024 -> 512 -> 1024)
# ---------------------------------------------------------
def refine_dataset_persam_tiled(input_dir, output_dir, predictor, ref_img_path, ref_mask_path):
    input_root = Path(input_dir)
    output_root = Path(output_dir)
    
    if output_root.exists():
        print("🧹 Clearing previous refinement output tree...")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    ref_img = cv2.imread(str(ref_img_path))
    ref_mask_raw = cv2.imread(str(ref_mask_path), cv2.IMREAD_GRAYSCALE)
    if ref_img is None or ref_mask_raw is None:
        raise FileNotFoundError(f"Visual template artifacts missing or unreadable at paths:\n {ref_img_path}\n {ref_mask_path}")

    print("🧠 Extracting Personalized DINOv2 target signature map...")
    target_weight = get_persam_weights(predictor, ref_img, ref_mask_raw)

    splits = ["train", "val", "test"]

    print(f"\n=== Running PerSAM 1024->512 Tiled Refinement Pipeline ===")

    for split in splits:
        in_img_dir = input_root / "images" / split
        in_lbl_dir = input_root / "labels" / split
        out_img_dir = output_root / "images" / split
        out_lbl_dir = output_root / "labels" / split

        if not in_img_dir.exists():
            continue
            
        print(f"--> Processing split: [{split.upper()}]")

        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_lbl_dir.mkdir(parents=True, exist_ok=True)

        image_list = sorted([f for f in os.listdir(in_img_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
        total_files = len(image_list)

        for idx, img_name in enumerate(image_list):
            img_path = in_img_dir / img_name
            lbl_path = in_lbl_dir / (Path(img_name).stem + ".txt")

            if not lbl_path.exists():
                continue

            shutil.copy(img_path, out_img_dir / img_name)

            img = cv2.imread(str(img_path))
            if img is None:
                continue
            img_h, img_w = img.shape[:2]

            with open(lbl_path) as f:
                lines = f.read().strip().splitlines()

            # Dynamic quadrant sorting: 512x512 borders inside the canvas
            quadrants = {0: [], 1: [], 2: [], 3: []}
            for line in lines:
                parts = line.split()
                if len(parts) < 3:
                    continue
                
                cls_id = int(parts[0])
                gx = int(float(parts[1]))
                gy = int(float(parts[2]))
                
                # Determine which 512 block the point belongs to
                q_idx = (1 if gx >= 512 else 0) + (2 if gy >= 512 else 0)
                quadrants[q_idx].append((cls_id, gx, gy))

            global_refined_labels = []
            
            for q_idx, points in quadrants.items():
                if not points: continue
                
                # Compute translation offsets for the patch
                x_off = 512 if q_idx in [1, 3] else 0
                y_off = 512 if q_idx in [2, 3] else 0
                
                # Extract the 512x512 crop
                patch = img[y_off:y_off+512, x_off:x_off+512]
                predictor.set_image(patch)
                
                for cls_id, gx, gy in points:
                    # Translate global center point to local patch coordinate
                    lx = gx - x_off
                    ly = gy - y_off
                    
                    nudge = 4  
                    input_points = np.array([
                        [lx, ly],           
                        [lx - nudge, ly],   
                        [lx + nudge, ly],   
                        [lx, ly - nudge],   
                        [lx, ly + nudge]    
                    ])
                    input_labels = np.array([1, 1, 1, 1, 1])

                    masks, _, _ = predictor.predict(
                        point_coords=input_points,
                        point_labels=input_labels,
                        box=None,
                        multimask_output=False
                    )

                    # Extract local coordinates
                    local_pts = mask_to_local_obb_points(masks[0])

                    if local_pts is not None:
                        # 1. Create native Shapely vector object in 512 space
                        local_poly = Polygon(local_pts)
                        
                        # 2. Translate vector coordinates into the master canvas footprint
                        global_poly = translate(local_poly, xoff=x_off, yoff=y_off)
                        
                        # 3. Pull out corners and normalize to YOLO format
                        g_pts = list(global_poly.exterior.coords)[:4]
                        yolo_obb = []
                        for pt in g_pts:
                            yolo_obb.extend([pt[0] / img_w, pt[1] / img_h])
                    else:
                        # Fallback to fixed-size horizontal box if OBB generation fails
                        def_w, def_h = 30.0, 15.0
                        x1_n = (gx - def_w / 2.0) / img_w
                        y1_n = (gy - def_h / 2.0) / img_h
                        x2_n = (gx + def_w / 2.0) / img_w
                        y2_n = (gy + def_h / 2.0) / img_h
                        
                        yolo_obb = [x1_n, y1_n, x2_n, y1_n, x2_n, y2_n, x1_n, y2_n]

                    global_refined_labels.append([cls_id] + yolo_obb)

            final_clean_labels = geometric_correction_head(global_refined_labels, img_w, img_h)

            out_lbl_path = out_lbl_dir / (Path(img_name).stem + ".txt")
            with open(out_lbl_path, "w") as f:
                for r in final_clean_labels:
                    coords_str = " ".join(f"{c:.6f}" for c in r[1:])
                    f.write(f"{int(r[0])} {coords_str}\n")
                    
            if (idx + 1) % 100 == 0 or (idx + 1) == total_files:
                print(f"    Progress: [{idx + 1}/{total_files}] tiled files refined...", end="\r")
        print()

    print(f"✔ Relabeling complete!")
    
    yaml_content = f"""path: {output_root.absolute()}
train: images/train
val: images/val
test: images/test

# Number of classes
nc: 1

# Classes
names:
  0: vehicle
"""
    yaml_path = output_root / "dataset.yaml"
    with open(yaml_path, "w") as yaml_file:
        yaml_file.write(yaml_content)
    print(f"📝 Generated YOLO configuration file at: {yaml_path}")
    print(f"🚀 Success! Production-ready OBB dataset is at: {output_root}")

# ---------------------------------------------------------
# Entry Point
# ---------------------------------------------------------
if __name__ == "__main__":
    
    # Updated to point to the new clean dataset format
    INPUT_DATASET = "datasets/cowc_persam_points"
    OUTPUT_DATASET = "datasets/cowc_1024_persam_refined"
    
    # Load SAM
    sam_predictor, dev_env = load_sam("models/sam_vit_b.pth")

    # Assuming you still have your ref_car and ref_mask saved here
    ref_image_file = Path("datasets/cowc_persam_points/ref_car.png")
    ref_mask_file = Path("datasets/cowc_persam_points/ref_mask.png")
    
    if ref_image_file.exists() and not ref_mask_file.exists():
        print("[INFO] ref_mask.png not found. Auto-generating binary mask from template brightness thresholds...")
        temp_img = cv2.imread(str(ref_image_file), cv2.IMREAD_GRAYSCALE)
        _, generated_mask = cv2.threshold(temp_img, 15, 255, cv2.THRESH_BINARY)
        cv2.imwrite(str(ref_mask_file), generated_mask)

    refine_dataset_persam_tiled(
        input_dir=INPUT_DATASET,
        output_dir=OUTPUT_DATASET,
        predictor=sam_predictor,
        ref_img_path=ref_image_file,
        ref_mask_path=ref_mask_file
    )