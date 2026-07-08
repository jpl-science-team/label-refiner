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
# 2. Mask → Local 512 Pixel Coordinates
# ---------------------------------------------------------
def mask_to_local_obb_points(mask):
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
# 4. Model Instantiation Utilities
# ---------------------------------------------------------
def load_sam(model_path="models/sam_vit_b.pth"):
    print(f"Loading SAM engine checkpoint: {model_path}")
    sam = sam_model_registry["vit_b"](checkpoint=model_path)
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    sam.to(device)
    predictor = SamPredictor(sam)
    return predictor

# ---------------------------------------------------------
# 5. Hybrid Tiled Inference Processing Loop
# ---------------------------------------------------------
def refine_dataset_dinov2_persam(input_dir, output_dir, predictor, ref_img_path, ref_mask_path):
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

    # ---------------------------------------------------------
    # ROBUST INPUT DETECTION & STRICT YOLO OUTPUT FORCING
    # ---------------------------------------------------------
    available_splits = [s for s in ["train", "val", "test"] if (input_root / s).exists()]
    
    if not available_splits:
        print(f"❌ Error: Could not find train or val folders directly inside {input_root}.")
        return

    print(f"\n=== Running Refinement & Forcing YOLO Format ===")
    print(f"Detected input splits: {available_splits}")

    for split in available_splits:
        split_in_dir = input_root / split
        
        # Check if the user's input has 'images' and 'labels' inside the split, 
        # or if the files are just sitting naked in the split folder.
        if (split_in_dir / "images").exists():
            in_img_dir = split_in_dir / "images"
            in_lbl_dir = split_in_dir / "labels" if (split_in_dir / "labels").exists() else in_img_dir
        else:
            in_img_dir = split_in_dir
            in_lbl_dir = split_in_dir

        # STRICT YOLO OUTPUT DIRECTORY CREATION
        out_img_dir = output_root / split / "images"
        out_lbl_dir = output_root / split / "labels"
            
        print(f"\n--> Slicing split partition: [{split.upper()}]")
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_lbl_dir.mkdir(parents=True, exist_ok=True)

        image_list = sorted([f for f in os.listdir(in_img_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
        total_files = len(image_list)

        if total_files == 0:
            print(f"    ⚠️ Warning: No images found in {in_img_dir}")
            continue

        for idx, img_name in enumerate(image_list):
            img_path = in_img_dir / img_name
            lbl_path = in_lbl_dir / (Path(img_name).stem + ".txt")

            if not lbl_path.exists():
                continue

            shutil.copy(img_path, out_img_dir / img_name)
            img = cv2.imread(str(img_path))
            if img is None: continue
            img_h, img_w = img.shape[:2]

            with open(lbl_path) as f:
                lines = f.read().strip().splitlines()

            quadrants = {0: [], 1: [], 2: [], 3: []}
            for line in lines:
                parts = line.split()
                if len(parts) < 3: continue
                
                cls_id = int(parts[0])
                gx = int(float(parts[1]) * img_w)
                gy = int(float(parts[2]) * img_h)
                
                q_idx = (1 if gx >= 512 else 0) + (2 if gy >= 512 else 0)
                quadrants[q_idx].append((cls_id, gx, gy))

            global_refined_labels = []
            
            for q_idx, points in quadrants.items():
                if not points: continue
                
                x_off = 512 if q_idx in [1, 3] else 0
                y_off = 512 if q_idx in [2, 3] else 0
                
                patch = img[y_off:y_off+512, x_off:x_off+512]
                
                patch_embedding = dino.extract_embedding(patch)
                ref_norm = F.normalize(ref_embedding, p=2, dim=1)
                patch_norm = F.normalize(patch_embedding, p=2, dim=1)
                sim_map = (patch_norm * ref_norm).sum(dim=1).squeeze(0)
                
                predictor.set_image(patch)
                
                for cls_id, gx, gy in points:
                    lx = gx - x_off
                    ly = gy - y_off
                    
                    nudge = 4  
                    input_points = np.array([
                        [lx, ly], [lx - nudge, ly], [lx + nudge, ly], [lx, ly - nudge], [lx, ly + nudge]    
                    ])
                    input_labels = np.array([1, 1, 1, 1, 1])

                    masks, _, _ = predictor.predict(
                        point_coords=input_points,
                        point_labels=input_labels,
                        box=None,
                        multimask_output=False
                    )

                    local_pts = mask_to_local_obb_points(masks[0])

                    if local_pts is not None:
                        local_poly = Polygon(local_pts)
                        global_poly = translate(local_poly, xoff=x_off, yoff=y_off)
                        g_pts = list(global_poly.exterior.coords)[:4]
                        yolo_obb = []
                        for pt in g_pts:
                            yolo_obb.extend([pt[0] / img_w, pt[1] / img_h])
                    else:
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
                    
            if (idx + 1) % 50 == 0 or (idx + 1) == total_files:
                print(f"    Progress: [{idx + 1}/{total_files}] images processed...", end="\r")
        print()

    # ---------------------------------------------------------
    # STRICT DATASET.YAML GENERATOR
    # ---------------------------------------------------------
    yaml_content = f"path: {output_root.absolute()}\n"
    
    if "train" in available_splits:
        yaml_content += "train: train/images\n"
    if "val" in available_splits:
        yaml_content += "val: val/images\n"
    if "test" in available_splits:
        yaml_content += "test: test/images\n"
        
    yaml_content += """
nc: 1
names:
  0: vehicle
"""
    with open(output_root / "dataset.yaml", "w") as f:
        f.write(yaml_content)
        
    print(f"\n🚀 Success! YOLO OBB dataset generated with strict folder separation at: {output_root}")

# ---------------------------------------------------------
# Entry Point Execution
# ---------------------------------------------------------
if __name__ == "__main__":
    INPUT_DATASET = "datasets/20260630_COWC_1024"
    OUTPUT_DATASET = "datasets/20260630_COWC_1024_DINOv2_Refined"
    
    sam_predictor = load_sam("models/sam_vit_b.pth")

    ref_image_file = Path("datasets/cowc_persam_points/ref_car.png")
    ref_mask_file = Path("datasets/cowc_persam_points/ref_mask.png")
    
    if not ref_image_file.exists():
        ref_image_file.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(ref_image_file), np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
        cv2.imwrite(str(ref_mask_file), np.ones((64, 64), dtype=np.uint8) * 255)

    refine_dataset_dinov2_persam(
        input_dir=INPUT_DATASET,
        output_dir=OUTPUT_DATASET,
        predictor=sam_predictor,
        ref_img_path=ref_image_file,
        ref_mask_path=ref_mask_file
    )