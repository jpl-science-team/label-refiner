#!/usr/bin/env python3
import cv2 
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path
import os
import time

# Import your existing pipeline tools
from VS_PerSAM_DINOv2 import DINOv2FeatureExtractor, load_sam, mask_to_yolo_obb

def read_all_target_cars(img_path, label_path):
    """
    Reads the specified YOLO label file and extracts ALL car coordinates.
    Handles both normalized (0.0-1.0) and absolute pixel coordinates.
    """
    label_path = Path(label_path)
    targets = []
    
    if not label_path.exists():
        print(f"⚠️ Warning: No label file found at {label_path}.")
        return targets
    
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"❌ Error: Could not read image at {img_path}")
        return targets
    img_h, img_w = img.shape[:2]

    with open(label_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 3:
                x_val = float(parts[1])
                y_val = float(parts[2])
                
                # Check if coordinates are normalized or absolute
                if x_val <= 2.0 and y_val <= 2.0:
                    targets.append((int(x_val * img_w), int(y_val * img_h)))
                else:
                    targets.append((int(x_val), int(y_val)))
                    
    return targets

def generate_stakeholder_visual(img_path, ref_img_path, ref_mask_path, targets, output_path="slide5_pipeline_visual.png"):
    """
    Runs all targets through the pipeline and generates a high-contrast infographic.
    """
    start_time = time.time()
    
    if not targets:
        print("❌ No targets found. Exiting.")
        return

    print(f"🎯 Found {len(targets)} targets in label file. Processing...")
    
    img = cv2.imread(str(img_path))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_h, img_w = img.shape[:2]

    ref_img = cv2.imread(str(ref_img_path))
    ref_mask_raw = cv2.imread(str(ref_mask_path), cv2.IMREAD_GRAYSCALE)

    # 1. Initialize Engines
    dino = DINOv2FeatureExtractor()
    predictor = load_sam("models/sam_vit_b.pth")

    print("Extracting DINO features...")
    ref_embedding = dino.extract_embedding(ref_img, ref_mask_raw)
    img_embedding = dino.extract_embedding(img)
    
    ref_norm = F.normalize(ref_embedding, p=2, dim=1)
    img_norm = F.normalize(img_embedding, p=2, dim=1)
    sim_map = (img_norm * ref_norm).sum(dim=1) 
    
    sim_map_resized = F.interpolate(
        sim_map.unsqueeze(0), size=(img_h, img_w), mode="bilinear", align_corners=False
    ).squeeze().cpu().numpy()

    # Normalize Heatmap so it's always bright
    sim_min, sim_max = sim_map_resized.min(), sim_map_resized.max()
    sim_norm_bright = (sim_map_resized - sim_min) / (sim_max - sim_min + 1e-8)

    print("Prompting SAM for all targets...")
    predictor.set_image(img_rgb)
    
    all_masks = []
    all_obbs = []
    
    for (tx, ty) in targets:
        input_points = np.array([[tx, ty]])
        input_labels = np.array([1])
        
        masks, _, _ = predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            multimask_output=False
        )
        mask = masks[0]
        obb = mask_to_yolo_obb(mask, img_w, img_h)
        
        all_masks.append(mask)
        if obb:
            all_obbs.append(obb)

    print("Generating Cropped Visuals...")
    
    # ==========================================
    # DYNAMIC CROPPING (Fit all cars in view)
    # ==========================================
    x_coords = [t[0] for t in targets]
    y_coords = [t[1] for t in targets]
    
    padding = 150
    x_min = max(0, min(x_coords) - padding)
    x_max = min(img_w, max(x_coords) + padding)
    y_min = max(0, min(y_coords) - padding)
    y_max = min(img_h, max(y_coords) + padding)

    # Crop all data layers
    img_crop = img_rgb[y_min:y_max, x_min:x_max].copy()
    heat_crop = sim_norm_bright[y_min:y_max, x_min:x_max]

    fig, axs = plt.subplots(1, 4, figsize=(24, 6))
    fig.patch.set_facecolor('white')

    # Panel 1: Original with Center Points
    axs[0].imshow(img_crop)
    for (tx, ty) in targets:
        crop_tx, crop_ty = tx - x_min, ty - y_min
        axs[0].scatter(crop_tx, crop_ty, color='yellow', s=200, alpha=0.5) 
        axs[0].scatter(crop_tx, crop_ty, color='red', s=80, edgecolors='white', linewidth=2, zorder=5)
    axs[0].set_title("1. Original Input (Center Points)", fontsize=18, fontweight='bold', pad=15)
    axs[0].axis('off')

    # Panel 2: DINOv2 Heatmap Overlay
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heat_crop), cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    overlay_heat = cv2.addWeighted(img_crop, 0.3, heatmap_rgb, 0.7, 0)
    axs[1].imshow(overlay_heat)
    axs[1].set_title("2. DINOv2 Feature Similarity", fontsize=18, fontweight='bold', pad=15)
    axs[1].axis('off')

    # Panel 3: SAM Mask Overlay (Neon Magenta)
    overlay_mask = img_crop.copy()
    for mask in all_masks:
        mask_crop = mask[y_min:y_max, x_min:x_max]
        overlay_mask[mask_crop] = [255, 0, 255] # Solid Neon Magenta
        
        # Add a bright cyan border
        contours, _ = cv2.findContours(mask_crop.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay_mask, contours, -1, (0, 255, 255), 2) 
        
    axs[2].imshow(overlay_mask)
    axs[2].set_title("3. Segment Anything (SAM) Masks", fontsize=18, fontweight='bold', pad=15)
    axs[2].axis('off')

    # Panel 4: Final OBBs (Neon Green)
    final_img = img_crop.copy()
    for obb in all_obbs:
        pts = (np.array(obb).reshape(4, 2) * [img_w, img_h]).astype(np.int32)
        pts[:, 0] -= x_min
        pts[:, 1] -= y_min
        cv2.polylines(final_img, [pts], isClosed=True, color=(0, 255, 0), thickness=4)
        
    axs[3].imshow(final_img)
    axs[3].set_title("4. Final Bounding Box Outputs", fontsize=18, fontweight='bold', pad=15)
    axs[3].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"✅ Success! Stakeholder visualization saved to {output_path}")
    print(f"⏱️ Pipeline execution time for this single image: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    
    TEST_IMAGE = "datasets/cowc_persam_points/images/train/Utah_AGRC_12TVL180140_y11264_x8192.png"
    TEST_LABEL = "datasets/cowc_persam_points/labels/train/Utah_AGRC_12TVL180140_y11264_x8192.txt" # UPDATE THIS TO YOUR EXACT PATH
    
    REF_IMAGE = "datasets/vedai_1024/vedai_color_centerpoint/ref_car.png"
    REF_MASK = "datasets/vedai_1024/vedai_color_centerpoint/ref_mask.png"
    
    # Extract ALL targets directly from the provided label path
    all_targets = read_all_target_cars(TEST_IMAGE, TEST_LABEL)
    
    generate_stakeholder_visual(
        img_path=TEST_IMAGE, 
        ref_img_path=REF_IMAGE, 
        ref_mask_path=REF_MASK, 
        targets=all_targets
    )