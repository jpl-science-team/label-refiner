#!/usr/bin/env python3

import os
import cv2 
import shutil
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from segment_anything import sam_model_registry, SamPredictor
from shapely.geometry import Polygon

# ---------------------------------------------------------
# 1. DINOv2 Feature Extractor & Prompt Refinement
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
# 2. Mask → Normalized YOLO OBB Converter (With Filtering Criteria)
# ---------------------------------------------------------
def mask_to_yolo_obb(mask, img_w, img_h):
    """Converts SAM binary mask to an 8-point OBB. Drops invalid/out-of-bounds masks."""
    contours, _ = cv2.findContours(mask.astype("uint8"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    cnt = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(cnt)
    
    (cx, cy), (w, h), angle = rect
    max_dimension = max(w, h)
    min_dimension = min(w, h)
    
    # HARD CRITERIA FILTER 1: Size limits for vehicles
    if max_dimension > 95 or max_dimension < 12: 
        return None
        
    # HARD CRITERIA FILTER 2: Aspect ratio limits
    aspect_ratio = max_dimension / (min_dimension + 1e-6)
    if aspect_ratio > 3.5 or aspect_ratio < 1.0:
        return None

    box_points = cv2.boxPoints(rect)
    pts = np.array(box_points, dtype="float32")
    
    # Order points: TL, TR, BR, BL
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
# 3. OBB Refinement & Filtering Stage
# ---------------------------------------------------------
def filter_obb_labels(refined_labels, img_w, img_h):
    """Refinement stage: Filters candidate OBB labels and drops bad geometries."""
    if len(refined_labels) == 0:
        return [], 0

    valid_labels = []
    dropped_count = 0

    for label in refined_labels:
        cls_id = label[0]
        pts = np.array(label[1:]).reshape(4, 2)
        rect = cv2.minAreaRect((pts * np.array([img_w, img_h])).astype(np.float32))
        (cx, cy), (w, h), angle = rect
        max_dim = max(w, h)
        min_dim = min(w, h)
        aspect = max_dim / (min_dim + 1e-6)
        
        # Criteria checks on converted OBB
        is_bad_box = max_dim > 95 or max_dim < 12 or aspect > 3.5

        if is_bad_box:
            dropped_count += 1
            # DROP: Do not save or adjust bad box
            continue
            
        valid_labels.append(label)

    return valid_labels, dropped_count

# ---------------------------------------------------------
# 4. Duplicate Box Calculator
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
# 6. Main Pipeline Strategy: DINO -> SAM -> Refine OBB Filter
# ---------------------------------------------------------
def run_dino_sam_obb_pipeline(input_dir, output_dir, predictor, ref_img_path, ref_mask_path):
    input_root = Path(input_dir)
    output_root = Path(output_dir)
    
    if output_root.exists():
        print("🧹 Flushing old processed artifacts...")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    ref_img = cv2.imread(str(ref_img_path))
    ref_mask_raw = cv2.imread(str(ref_mask_path), cv2.IMREAD_GRAYSCALE)
    if ref_img is None or ref_mask_raw is None:
        raise FileNotFoundError("Missing one-shot reference template target elements.")

    # 1. Initialize DINO
    dino = DINOv2FeatureExtractor()
    print("✨ Extracting DINO reference features...")
    ref_embedding = dino.extract_embedding(ref_img, ref_mask_raw)

    metrics_tracker = {
        "total_targets_processed": 0,
        "sam_mask_failures": 0,
        "dropped_geometric": 0,
        "total_dropped": 0,
        "duplicate_boxes": 0,
        "dino_similarities": []
    }

    if (input_root / "images").exists():
        available_splits = ["."]
        has_subsplits = False
    else:
        available_splits = [s for s in ["train", "val", "test"] if (input_root / s).exists()]
        has_subsplits = True

    print(f"\n=== Executing DINO -> SAM -> OBB Refinement Pipeline ===")

    for split in available_splits:
        split_in_dir = input_root / split if split != "." else input_root
        
        in_img_dir = split_in_dir / "images" if (split_in_dir / "images").exists() else split_in_dir
        in_lbl_dir = split_in_dir / "labels" if (split_in_dir / "labels").exists() else split_in_dir

        out_img_dir = output_root / split / "images" if has_subsplits else output_root / "images"
        out_lbl_dir = output_root / split / "labels" if has_subsplits else output_root / "labels"
            
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

            # DINO Feature Extraction & Similarity Mapping
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

            # Set current image for SAM
            predictor.set_image(img)
            candidate_labels = []

            with open(lbl_path) as f:
                lines = f.read().strip().splitlines()

            for line in lines:
                parts = line.split()
                if len(parts) < 3: continue
                
                cls_id = int(parts[0])
                gx = int(float(parts[1]) * img_w)
                gy = int(float(parts[2]) * img_h)
                
                metrics_tracker["total_targets_processed"] += 1

                # STEP 1: DINO Prompt Search
                search_radius = 6
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

                prof_radius = 24 
                px_min, px_max = max(0, best_gx - prof_radius), min(img_w, best_gx + prof_radius)
                py_min, py_max = max(0, best_gy - prof_radius), min(img_h, best_gy + prof_radius)
                
                local_profile = sim_map_resized[py_min:py_max, px_min:px_max].cpu().numpy()
                threshold = 0.65 * (similarity_score + 1e-6)
                binary_profile = local_profile > threshold
                
                y_indices, x_indices = np.where(binary_profile)
                if len(x_indices) > 0 and len(y_indices) > 0:
                    dynamic_radius_x = min(max(12, int((x_indices.max() - x_indices.min()) / 2) + 1), 18)
                    dynamic_radius_y = min(max(12, int((y_indices.max() - y_indices.min()) / 2) + 1), 18)
                else:
                    dynamic_radius_x, dynamic_radius_y = 15, 15

                input_box = np.array([
                    best_gx - dynamic_radius_x, 
                    best_gy - dynamic_radius_y, 
                    best_gx + dynamic_radius_x, 
                    best_gy + dynamic_radius_y
                ])

                rx = dynamic_radius_x + 1
                ry = dynamic_radius_y + 1
                input_coords = [
                    [best_gx, best_gy],
                    [best_gx - rx, best_gy],
                    [best_gx + rx, best_gy],
                    [best_gx, best_gy - ry],
                    [best_gx, best_gy + ry],
                    [best_gx - rx, best_gy - ry],
                    [best_gx + rx, best_gy - ry],
                    [best_gx - rx, best_gy + ry],
                    [best_gx + rx, best_gy + ry]
                ]
                input_points = np.array(input_coords)
                input_labels = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0])

                # STEP 2: Feed Prompts to SAM
                masks, _, _ = predictor.predict(
                    point_coords=input_points,
                    point_labels=input_labels,
                    box=input_box[None, :], 
                    multimask_output=False
                )

                # STEP 3: Mask to OBB & Filter check
                yolo_obb = mask_to_yolo_obb(masks[0], img_w, img_h)

                if yolo_obb is None:
                    metrics_tracker["sam_mask_failures"] += 1
                    metrics_tracker["total_dropped"] += 1
                    continue

                candidate_labels.append([cls_id] + yolo_obb)

            # STEP 4: OBB Refinement / Final Filtering Stage
            final_clean_labels, dropped_geom = filter_obb_labels(candidate_labels, img_w, img_h)
            metrics_tracker["dropped_geometric"] += dropped_geom
            metrics_tracker["total_dropped"] += dropped_geom
            
            dups = count_duplicate_boxes(final_clean_labels, img_w, img_h)
            metrics_tracker["duplicate_boxes"] += dups

            out_lbl_path = out_lbl_dir / (Path(img_name).stem + ".txt")
            with open(out_lbl_path, "w") as f:
                for r in final_clean_labels:
                    coords_str = " ".join(f"{c:.6f}" for c in r[1:])
                    f.write(f"{int(r[0])} {coords_str}\n")
                    
            if (idx + 1) % 50 == 0 or (idx + 1) == total_files:
                print(f"    Progress: [{idx + 1}/{total_files}] images processed...", end="\r")
        print()

    # Diagnostics report
    total_inst = metrics_tracker["total_targets_processed"]
    dropped = metrics_tracker["total_dropped"]
    avg_dino = np.mean(metrics_tracker["dino_similarities"]) if metrics_tracker["dino_similarities"] else 0.0
    
    dropped_pct = (dropped / total_inst) * 100 if total_inst > 0 else 0.0
    retained_pct = 100.0 - dropped_pct

    report_path = output_root / "pipeline_refinement_metrics.txt"
    with open(report_path, "w") as f:
        f.write("=" * 65 + "\n")
        f.write("      DINO -> SAM -> OBB REFINEMENT EVALUATION\n")
        f.write("=" * 65 + "\n")
        f.write(f"Total Candidate Targets Fed     : {total_inst}\n")
        f.write(f"Total Duplicate Boxes Dropped   : {metrics_tracker['duplicate_boxes']}\n")
        f.write(f"Total Filtered / Dropped Boxes  : {dropped} ({dropped_pct:.2f}%)\n")
        f.write(f"  - SAM Mask / Filter Failures  : {metrics_tracker['sam_mask_failures']}\n")
        f.write(f"  - OBB Refinement Filter Drops : {metrics_tracker['dropped_geometric']}\n")
        f.write("-" * 65 + "\n")
        f.write(f"Final Retained Label Rate       : {retained_pct:.2f}%\n")
        f.write(f"Mean DINO Similarity Score      : {avg_dino:.4f}\n")
        f.write("=" * 65 + "\n")

    print(f"\n📊 Diagnostic metrics exported to: {report_path}")

if __name__ == "__main__":
    INPUT_DATASET = "datasets/vedai_1024/vedai_color_centerpoint"
    OUTPUT_DATASET = "datasets/vedai_1024/vedai_color_OBB_Refined"
    
    sam_predictor = load_sam("models/sam_vit_b.pth")

    ref_image_file = Path("datasets/vedai_1024/vedai_color_centerpoint/ref_car.png")
    ref_mask_file = Path("datasets/vedai_1024/vedai_color_centerpoint/ref_mask.png")
    
    if ref_image_file.exists() and not ref_mask_file.exists():
        temp_img = cv2.imread(str(ref_image_file), cv2.IMREAD_GRAYSCALE)
        _, generated_mask = cv2.threshold(temp_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cv2.imwrite(str(ref_mask_file), generated_mask)
        
    run_dino_sam_obb_pipeline(
        input_dir=INPUT_DATASET,
        output_dir=OUTPUT_DATASET,
        predictor=sam_predictor,
        ref_img_path=ref_image_file,
        ref_mask_path=ref_mask_file
    )