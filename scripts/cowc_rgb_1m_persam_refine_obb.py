#!/usr/bin/env python3

import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from segment_anything import sam_model_registry, SamPredictor

# ---------------------------------------------------------
# 1. PerSAM Visual Weights Extractor
# ---------------------------------------------------------
def get_persam_weights(predictor, ref_image, ref_mask):
    """
    Extracts DINOv2 visual features from the reference template 
    so SAM recognizes the custom styling and texture of the cars.
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
# 2. Mask → Normalized YOLO OBB Converter (With Geometric Filtering)
# ---------------------------------------------------------
def mask_to_yolo_obb(mask, img_w, img_h):
    """
    Converts binary masks into a 4-point structural Oriented Bounding Box (OBB).
    Includes defensive filtering to reject giant hallucinations and noise.
    Adjusted for 1m/pixel spatial resolution.
    """
    contours, _ = cv2.findContours(mask.astype("uint8"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    cnt = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(cnt)
    
    # Extract structural dimensions to run verification checks
    (cx, cy), (w, h), angle = rect
    max_dimension = max(w, h)
    min_dimension = min(w, h)
    
    # ---------------------------------------------------------
    # DEFENSIVE FILTERING: Scaled for 1m GSD (1 pixel ≈ 1 meter)
    # ---------------------------------------------------------
    if max_dimension > 12:  # Reject boxes > 12 meters
        return None
        
    if max_dimension < 2:   # Reject boxes < 2 meters
        return None
        
    aspect_ratio = max_dimension / (min_dimension + 1e-6)
    if aspect_ratio > 4.5 or aspect_ratio < 1.1:
        return None
    # ---------------------------------------------------------

    box_points = cv2.boxPoints(rect)
    
    pts = np.array(box_points, dtype="float32")
    x_sorted = pts[np.argsort(pts[:, 0]), :]
    left_pts = x_sorted[:2, :]
    right_pts = x_sorted[2:, :]
    
    tl = left_pts[np.argsort(left_pts[:, 1])[0]]
    bl = left_pts[np.argsort(left_pts[:, 1])[1]]
    tr = right_pts[np.argsort(right_pts[:, 1])[0]]
    br = right_pts[np.argsort(right_pts[:, 1])[1]]
    
    return [
        tl[0] / img_w, tl[1] / img_h,
        tr[0] / img_w, tr[1] / img_h,
        br[0] / img_w, br[1] / img_h,
        bl[0] / img_w, bl[1] / img_h
    ]

# ---------------------------------------------------------
# 3. Post-Processing Geometric Correction Head
# ---------------------------------------------------------
def geometric_correction_head(refined_labels, img_w, img_h):
    """
    STRICT POST-PROCESSING HEAD: Resolves remaining edge case failures.
    Adjusted dimensions for 1m/pixel resolution.
    """
    if len(refined_labels) == 0:
        return refined_labels

    corrected_labels = []
    valid_cars = []

    # First pass: find valid cars to act as size anchors
    for label in refined_labels:
        cls_id = label[0]
        pts = np.array(label[1:]).reshape(4, 2)
        
        rect = cv2.minAreaRect((pts * np.array([img_w, img_h])).astype(np.float32))
        (cx, cy), (w, h), angle = rect
        max_dim = max(w, h)
        min_dim = min(w, h)
        aspect = max_dim / (min_dim + 1e-6)

        # Valid car criteria (3 to 10 meters/pixels)
        if 3 < max_dim < 10 and 1.3 < aspect < 3.2 and angle != 0.0:
            valid_cars.append({
                'center': (cx, cy),
                'size': (w, h),
                'angle': angle,
                'raw_label': label
            })

    # Second pass: fix the bad boxes
    for label in refined_labels:
        cls_id = label[0]
        pts = np.array(label[1:]).reshape(4, 2)
        rect = cv2.minAreaRect((pts * np.array([img_w, img_h])).astype(np.float32))
        (cx, cy), (w, h), angle = rect
        max_dim = max(w, h)
        
        # Identify bad boxes based on 1m scale
        is_bad_box = max_dim > 12 or max_dim < 2 or angle == 0.0 or angle == 90.0

        if is_bad_box:
            if len(valid_cars) > 0:
                distances = [np.linalg.norm(np.array((cx, cy)) - np.array(vc['center'])) for vc in valid_cars]
                closest_car = valid_cars[np.argmin(distances)]
                matched_w, matched_h = closest_car['size']
                matched_angle = closest_car['angle']
            else:
                # Fallback to standard car size: 4.5m length x 2.0m width
                matched_w, matched_h = 4.5, 2.0
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
# 5. Main Datasets Processing Loop
# ---------------------------------------------------------
def refine_dataset_persam(input_root, output_root, predictor, ref_img_path, ref_mask_path):
    input_root = Path(input_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    ref_img = cv2.imread(str(ref_img_path))
    ref_mask_raw = cv2.imread(str(ref_mask_path), cv2.IMREAD_GRAYSCALE)
    if ref_img is None or ref_mask_raw is None:
        raise FileNotFoundError(f"Visual template artifacts missing or unreadable at paths:\n {ref_img_path}\n {ref_mask_path}")

    print("Extracting Personalized DINOv2 target signature map...")
    target_weight = get_persam_weights(predictor, ref_img, ref_mask_raw)

    datasets = ["cowc_rgb_1m"]

    for dataset in datasets:
        print(f"\n=== Running PerSAM Refinement: {dataset} ===")

        in_img_dir = input_root / dataset / "images"
        in_lbl_dir = input_root / dataset / "labels"
        out_img_dir = output_root / dataset / "images"
        out_lbl_dir = output_root / dataset / "labels"

        if not in_img_dir.exists():
            print(f"[WARN] Directory {in_img_dir} does not exist, skipping dataset.")
            continue

        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_lbl_dir.mkdir(parents=True, exist_ok=True)

        image_list = sorted([f for f in os.listdir(in_img_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))])

        for img_name in image_list:
            img_path = in_img_dir / img_name
            lbl_path = in_lbl_dir / (Path(img_name).stem + ".txt")

            if not lbl_path.exists():
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                continue
            img_h, img_w = img.shape[:2]

            cv2.imwrite(str(out_img_dir / img_name), img)

            with open(lbl_path) as f:
                lines = f.read().strip().splitlines()

            refined_labels = []
            predictor.set_image(img)

            img_features = predictor.get_image_embedding()
            pooled_target_weight = target_weight.mean(dim=(-1, -2), keepdim=True)
            
            similarity_map = torch.mean(img_features * pooled_target_weight, dim=1, keepdim=True)
            similarity_map = F.interpolate(similarity_map, size=(img_h, img_w), mode="bilinear").squeeze().cpu().numpy()

            for line in lines:
                parts = list(map(float, line.split()))
                if len(parts) < 5:
                    continue
                cls_id, cx, cy, w, h = parts[:5]

                px = int(round(cx * img_w))
                py = int(round(cy * img_h))

                # 1. Use ONLY the exact center point
                input_point = np.array([[px, py]])
                input_label = np.array([1])

                # 2. Trap SAM inside a strict 6x6 pixel boundary box
                box_size = 6  
                box_prompt = np.array([
                    max(0, px - (box_size / 2.0)), 
                    max(0, py - (box_size / 2.0)), 
                    min(img_w - 1, px + (box_size / 2.0)), 
                    min(img_h - 1, py + (box_size / 2.0))
                ])

                # 3. Pass the box prompt to force a hard cutoff between parked cars
                masks, scores, _ = predictor.predict(
                    point_coords=input_point,
                    point_labels=input_label,
                    box=box_prompt,
                    multimask_output=False
                )

                mask = masks[0]
                yolo_obb = mask_to_yolo_obb(mask, img_w, img_h)

                if yolo_obb is None:
                    x1_n = cx - (w / 2.0)
                    y1_n = cy - (h / 2.0)
                    x2_n = cx + (w / 2.0)
                    y2_n = cy + (h / 2.0)
                    
                    yolo_obb = [x1_n, y1_n, x2_n, y1_n, x2_n, y2_n, x1_n, y2_n]

                refined_labels.append([int(cls_id)] + yolo_obb)

            final_clean_labels = geometric_correction_head(refined_labels, img_w, img_h)

            out_lbl_path = out_lbl_dir / (Path(img_name).stem + ".txt")
            with open(out_lbl_path, "w") as f:
                for r in final_clean_labels:
                    coords_str = " ".join(f"{c:.6f}" for c in r[1:])
                    f.write(f"{int(r[0])} {coords_str}\n")

        print(f"✔ Dataset processing complete: {dataset}")

# ---------------------------------------------------------
# Entry Point
# ---------------------------------------------------------
if __name__ == "__main__":
    sam_predictor, dev_env = load_sam("models/sam_vit_b.pth")

    ref_image_file = Path("datasets/cowc_rgb_1m/ref_car.png")
    ref_mask_file = Path("datasets/cowc_rgb_1m/ref_mask.png")
    
    if ref_image_file.exists() and not ref_mask_file.exists():
        print("[INFO] ref_mask.png not found. Auto-generating binary mask from template brightness thresholds...")
        temp_img = cv2.imread(str(ref_image_file), cv2.IMREAD_GRAYSCALE)
        _, generated_mask = cv2.threshold(temp_img, 15, 255, cv2.THRESH_BINARY)
        cv2.imwrite(str(ref_mask_file), generated_mask)

    refine_dataset_persam(
        input_root="datasets/cowc_test_subset",
        output_root="datasets/cowc_test_refined",
        predictor=sam_predictor,
        ref_img_path=ref_image_file,
        ref_mask_path=ref_mask_file
    )