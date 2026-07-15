#!/usr/bin/env python3

import os
import cv2 
import shutil
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from segment_anything import sam_model_registry, SamPredictor

import shapely
from shapely.geometry import Polygon
from shapely.affinity import translate

# ---------------------------------------------------------
# 1. DINOv2 Global Feature Extractor Head
# ---------------------------------------------------------
class DINOv2FeatureExtractor:
    def __init__(self, model_name="dinov2_vits14"):
        print(f"🧠 Initializing self-supervised {model_name} backend engine...")
        self.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        self.model = torch.hub.load('facebookresearch/dinov2', model_name).to(self.device)
        self.model.eval()
        
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)

    @torch.no_grad()
    def extract_embedding(self, cv2_img, mask=None):
        img_rgb = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).float().to(self.device) / 255.0
        tensor = (tensor - self.mean) / self.std
        
        h, w = tensor.shape[-2:]
        new_h = ((h + 13) // 14) * 14
        new_w = ((w + 13) // 14) * 14
        if new_h != h or new_w != w:
            tensor = F.interpolate(tensor, size=(new_h, new_w), mode="bilinear", align_corners=False)

        features = self.model.forward_features(tensor)["x_norm_patchtokens"]
        patch_h, patch_w = new_h // 14, new_w // 14
        features = features.reshape(1, patch_h, patch_w, -1).permute(0, 3, 1, 2)
        
        if mask is not None:
            mask_tensor = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0).to(self.device)
            mask_resized = F.interpolate(mask_tensor, size=(patch_h, patch_w), mode="nearest")
            target_feat = features * mask_resized
            features = target_feat.sum(dim=(-1, -2)) / (mask_resized.sum(dim=(-1, -2)) + 1e-6)
            features = features.unsqueeze(-1).unsqueeze(-1)
            
        return features

# ---------------------------------------------------------
# 2. Mask → Normalized YOLO OBB Converter
# ---------------------------------------------------------
def mask_to_yolo_obb(mask, img_w, img_h):
    contours, _ = cv2.findContours(mask.astype("uint8"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    cnt = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(cnt)
    
    (cx, cy), (w, h), angle = rect
    max_dimension = max(w, h)
    min_dimension = min(w, h)
    
    # Relaxed constraints to accommodate small VEDAI vehicles
    if max_dimension > 150 or max_dimension < 4: 
        return None
        
    aspect_ratio = max_dimension / (min_dimension + 1e-6)
    if aspect_ratio > 5.0 or aspect_ratio < 1.0:
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
    if len(refined_labels) == 0:
        return refined_labels, 0

    corrected_labels = []
    valid_cars = []
    correction_count = 0

    for label in refined_labels:
        pts = np.array(label[1:]).reshape(4, 2)
        rect = cv2.minAreaRect((pts * np.array([img_w, img_h])).astype(np.float32))
        (cx, cy), (w, h), angle = rect
        max_dim = max(w, h)
        min_dim = min(w, h)
        aspect = max_dim / (min_dim + 1e-6)

        if 8 < max_dim < 100 and 1.0 <= aspect < 5.0:
            valid_cars.append({
                'center': (cx, cy),
                'size': (w, h),
                'angle': angle
            })

    for label in refined_labels:
        cls_id = label[0]
        pts = np.array(label[1:]).reshape(4, 2)
        rect = cv2.minAreaRect((pts * np.array([img_w, img_h])).astype(np.float32))
        (cx, cy), (w, h), angle = rect
        max_dim = max(w, h)
        min_dim = min(w, h)
        aspect = max_dim / (min_dim + 1e-6)
        
        is_bad_box = max_dim > 150 or max_dim < 4 or aspect > 5.0

        if is_bad_box:
            correction_count += 1
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

    return corrected_labels, correction_count

# ---------------------------------------------------------
# 4. Duplicate Box Calculator (IoU Metric)
# ---------------------------------------------------------
def count_duplicate_boxes(labels, img_w, img_h, iou_threshold=0.85):
    duplicates = 0
    n = len(labels)
    if n < 2: 
        return 0
        
    polys = []
    for lbl in labels:
        pts = np.array(lbl[1:]).reshape(4, 2) * np.array([img_w, img_h])
        p = Polygon(pts)
        if not p.is_valid: p = p.buffer(0)
        polys.append(p)
        
    to_skip = set()
    for i in range(n):
        if i in to_skip: continue
        for j in range(i+1, n):
            if j in to_skip: continue
            
            try:
                inter = polys[i].intersection(polys[j]).area
                union = polys[i].area + polys[j].area - inter
                iou = inter / (union + 1e-6)
                
                if iou >= iou_threshold:
                    duplicates += 1
                    to_skip.add(j)
            except Exception:
                pass
                
    return duplicates

# ---------------------------------------------------------
# 5. Load SAM Configuration
# ---------------------------------------------------------
def load_sam(model_path="models/sam_vit_b.pth"):
    print(f"Loading SAM engine checkpoint: {model_path}")
    sam = sam_model_registry["vit_b"](checkpoint=model_path)
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    sam.to(device)
    predictor = SamPredictor(sam)
    return predictor

# ---------------------------------------------------------
# 6. Main Pipeline
# ---------------------------------------------------------
def refine_dataset_dinov2(input_dir, output_dir, predictor, ref_img_path, ref_mask_path):
    input_root = Path(input_dir)
    output_root = Path(output_dir)
    
    if output_root.exists():
        print("🧹 Flushing old processed artifacts...")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    ref_img = cv2.imread(str(ref_img_path))
    ref_mask_raw = cv2.imread(str(ref_mask_path), cv2.IMREAD_GRAYSCALE)
    if ref_img is None or ref_mask_raw is None:
        raise FileNotFoundError(f"Missing one-shot reference template target elements.")

    dino = DINOv2FeatureExtractor()
    print("✨ Mapping targeted DINOv2 semantic one-shot template array...")
    ref_embedding = dino.extract_embedding(ref_img, ref_mask_raw)

    metrics_tracker = {
        "total_targets_processed": 0,
        "sam_mask_failures": 0,
        "geometric_corrections": 0,
        "total_reversions": 0,
        "duplicate_boxes": 0,
        "dino_similarities": []
    }

    if (input_root / "images").exists():
        available_splits = ["."]
        has_subsplits = False
    else:
        available_splits = [s for s in ["train", "val", "test"] if (input_root / s).exists()]
        has_subsplits = True

    print(f"\n=== Running DINOv2 + SAM Refinement Pipeline ===")

    for split in available_splits:
        split_in_dir = input_root / split if split != "." else input_root
        
        if (split_in_dir / "images").exists():
            in_img_dir = split_in_dir / "images"
            in_lbl_dir = split_in_dir / "labels" if (split_in_dir / "labels").exists() else split_in_dir
        else:
            in_img_dir = split_in_dir
            in_lbl_dir = split_in_dir

        if has_subsplits:
            out_img_dir = output_root / split / "images"
            out_lbl_dir = output_root / split / "labels"
        else:
            out_img_dir = output_root / "images"
            out_lbl_dir = output_root / "labels"
            
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_lbl_dir.mkdir(parents=True, exist_ok=True)

        image_list = sorted([f for f in os.listdir(in_img_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
        total_files = len(image_list)

        if total_files == 0: continue

        for idx, img_name in enumerate(image_list):
            img_path = in_img_dir / img_name
            lbl_path = in_lbl_dir / (Path(img_name).stem + ".txt")

            if not lbl_path.exists(): continue

            shutil.copy(img_path, out_img_dir / img_name)
            img = cv2.imread(str(img_path))
            if img is None: continue
            
            img_h, img_w = img.shape[:2]

            # 1. DINOv2 Whole Image Similarity Mapping
            img_embedding = dino.extract_embedding(img)
            ref_norm = F.normalize(ref_embedding, p=2, dim=1)
            img_norm = F.normalize(img_embedding, p=2, dim=1)
            sim_map = (img_norm * ref_norm).sum(dim=1) 
            
            sim_map_resized = F.interpolate(
                sim_map.unsqueeze(0), 
                size=(img_h, img_w), 
                mode="bilinear", 
                align_corners=False
            ).squeeze()

            # 2. SAM Preparation
            predictor.set_image(img)
            refined_labels = []

            with open(lbl_path) as f:
                lines = f.read().strip().splitlines()

            for line in lines:
                parts = line.split()
                if len(parts) < 3: continue
                
                cls_id = int(parts[0])
                
                gx = int(float(parts[1]) * img_w)
                gy = int(float(parts[2]) * img_h)
                
                metrics_tracker["total_targets_processed"] += 1
                
                # ==========================================================
                # TIGHT LEASH DINOv2 SHIFT LOGIC
                # Restrict search to a 4-pixel radius so DINO cannot step off the car
                # ==========================================================
                search_radius = 4
                x_min, x_max = max(0, gx - search_radius), min(img_w, gx + search_radius + 1)
                y_min, y_max = max(0, gy - search_radius), min(img_h, gy + search_radius + 1)
                
                window = sim_map_resized[y_min:y_max, x_min:x_max]
                
                if window.numel() > 0:
                    max_idx = torch.argmax(window.reshape(-1))
                    best_y, best_x = np.unravel_index(max_idx.cpu().numpy(), window.shape)
                    best_gx = x_min + best_x
                    best_gy = y_min + best_y
                    similarity_score = window[best_y, best_x].item()
                else:
                    best_gx, best_gy = gx, gy
                    similarity_score = 0.0

                metrics_tracker["dino_similarities"].append(similarity_score)

                if similarity_score < 0.03:
                    best_gx, best_gy = gx, gy

                # =======================================================
                # CRITICAL FIX: Prompt SAM with a tight Bounding Box
                # This traps SAM and prevents it from bleeding into the road
                # =======================================================
                box_radius = 12 # 24x24 pixel box (slightly larger than a VEDAI car)
                input_box = np.array([
                    best_gx - box_radius, 
                    best_gy - box_radius, 
                    best_gx + box_radius, 
                    best_gy + box_radius
                ])

                masks, _, _ = predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=input_box[None, :], 
                    multimask_output=False
                )

                yolo_obb = mask_to_yolo_obb(masks[0], img_w, img_h)

                if yolo_obb is None:
                    metrics_tracker["sam_mask_failures"] += 1
                    metrics_tracker["total_reversions"] += 1
                    def_w, def_h = 30.0, 15.0 # Fallback pixels
                    x1_n = (gx - def_w / 2.0) / img_w
                    y1_n = (gy - def_h / 2.0) / img_h
                    x2_n = (gx + def_w / 2.0) / img_w
                    y2_n = (gy + def_h / 2.0) / img_h
                    yolo_obb = [x1_n, y1_n, x2_n, y1_n, x2_n, y2_n, x1_n, y2_n]

                refined_labels.append([cls_id] + yolo_obb)

            # Post-Processing Head & Duplicates Tracker
            final_clean_labels, corrections = geometric_correction_head(refined_labels, img_w, img_h)
            metrics_tracker["geometric_corrections"] += corrections
            metrics_tracker["total_reversions"] += corrections
            
            dups = count_duplicate_boxes(final_clean_labels, img_w, img_h)
            metrics_tracker["duplicate_boxes"] += dups

            out_lbl_path = out_lbl_dir / (Path(img_name).stem + ".txt")
            with open(out_lbl_path, "w") as f:
                for r in final_clean_labels:
                    coords_str = " ".join(f"{c:.6f}" for c in r[1:])
                    f.write(f"{int(r[0])} {coords_str}\n")
                    
            if (idx + 1) % 50 == 0 or (idx + 1) == total_files:
                print(f"    Progress: [{idx + 1}/{total_files}] processed target assets...", end="\r")
        print()

    # ---------------------------------------------------------
    # EXPORT PIPELINE DIAGNOSTIC REPORT
    # ---------------------------------------------------------
    total_inst = metrics_tracker["total_targets_processed"]
    reversions = metrics_tracker["total_reversions"]
    duplicates = metrics_tracker["duplicate_boxes"]
    avg_dino = np.mean(metrics_tracker["dino_similarities"]) if metrics_tracker["dino_similarities"] else 0.0
    
    reversion_pct = (reversions / total_inst) * 100 if total_inst > 0 else 0.0
    success_pct = 100.0 - reversion_pct

    report_path = output_root / "pipeline_refinement_metrics.txt"
    with open(report_path, "w") as f:
        f.write("=" * 65 + "\n")
        f.write("      SAM + DINOv2 PSEUDO-LABEL REFINEMENT EXTRACTION STUDY\n")
        f.write("=" * 65 + "\n")
        f.write(f"Processed Dataset Target Root   : {output_root.name}\n")
        f.write(f"Source Reference Material Path  : {input_root.name}\n")
        f.write("-" * 65 + "\n")
        f.write(f"Total Object Center Points Fed  : {total_inst}\n")
        f.write(f"Total Duplicate Boxes Produced  : {duplicates}\n")
        f.write(f"Total Reversions to Default Box : {reversions} ({reversion_pct:.2f}% of data)\n")
        f.write(f"  - SAM Mask Failures           : {metrics_tracker['sam_mask_failures']}\n")
        f.write(f"  - Geometric Head Outliers     : {metrics_tracker['geometric_corrections']}\n")
        f.write("-" * 65 + "\n")
        f.write(f"Pipeline Extraction Success Rate: {success_pct:.2f}%\n")
        f.write(f"Mean DINOv2 Cosine Similarity   : {avg_dino:.4f}\n")
        f.write("=" * 65 + "\n")

    print(f"\n📊 Diagnostics report generated successfully at: {report_path}")

    # Generate YAML
    yaml_content = f"path: {output_root.absolute()}\n"
    if has_subsplits:
        if "train" in available_splits: yaml_content += "train: train/images\n"
        if "val" in available_splits: yaml_content += "val: val/images\n"
        if "test" in available_splits: yaml_content += "test: test/images\n"
    else:
        yaml_content += "train: images\nval: images\n"
        
    yaml_content += "\nnc: 1\nnames:\n  0: vehicle\n"
    with open(output_root / "dataset.yaml", "w") as f:
        f.write(yaml_content)
        
    print(f"🚀 Success! YOLO OBB dataset generated cleanly at: {output_root}")

if __name__ == "__main__":
    INPUT_DATASET = "datasets/vedai/vedai_color_centerpoint"
    OUTPUT_DATASET = "datasets/vedai/vedai_color_OBB_Refined"
    
    sam_predictor = load_sam("models/sam_vit_b.pth")

    ref_image_file = Path("datasets/vedai/vedai_color_centerpoint/ref_car.png")
    ref_mask_file = Path("datasets/vedai/vedai_color_centerpoint/ref_mask.png")
    
    if ref_image_file.exists() and not ref_mask_file.exists():
        print("[INFO] ref_mask.png not found. Auto-generating binary mask via Otsu's thresholding...")
        temp_img = cv2.imread(str(ref_image_file), cv2.IMREAD_GRAYSCALE)
        _, generated_mask = cv2.threshold(temp_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cv2.imwrite(str(ref_mask_file), generated_mask)
        
    refine_dataset_dinov2(
        input_dir=INPUT_DATASET,
        output_dir=OUTPUT_DATASET,
        predictor=sam_predictor,
        ref_img_path=ref_image_file,
        ref_mask_path=ref_mask_file
    )