#!/usr/bin/env python3

import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path
from segment_anything import sam_model_registry, SamPredictor

# ---------------------------------------------------------
# 1. Metric Calculation: Manual Labeling Reduction
# ---------------------------------------------------------
def calculate_manual_reduction(total_output_dir, needs_refinement_dir):
    """
    Counts images that passed vs. failed the pipeline to prove 
    the reduction in human-in-the-loop manual labeling.
    """
    total_output_path = Path(total_output_dir)
    needs_refinement_path = Path(needs_refinement_dir)
    
    if not total_output_path.exists() or not needs_refinement_path.exists():
        print("[WARN] Directories not found. Check your paths.")
        return
        
    # Count total processed images and those needing manual fix
    total_processed = len(list(total_output_path.glob("*.jpg"))) + len(list(total_output_path.glob("*.png")))
    needs_manual = len(list(needs_refinement_path.glob("*.jpg"))) + len(list(needs_refinement_path.glob("*.png")))
    
    if total_processed == 0:
        print("No images found in the output directory.")
        return

    automated_success = total_processed - needs_manual
    automation_rate = (automated_success / total_processed) * 100

    print("\n" + "="*50)
    print(" 📊 PIPELINE EFFICIENCY METRICS")
    print("="*50)
    print(f"Total Images Processed:    {total_processed}")
    print(f"Passed Auto-Labeling:      {automated_success}")
    print(f"Sent to Manual Review:     {needs_manual}")
    print(f"Manual Labeling Reduction: {automation_rate:.2f}%")
    print("="*50 + "\n")

# ---------------------------------------------------------
# 2. Similarity Map & Visualization Generator
# ---------------------------------------------------------
def generate_visuals(predictor, ref_img, ref_mask, test_img_path, labels_path, output_vis_path):
    """
    Generates the similarity map and plots a 1x4 Matplotlib grid 
    perfect for your presentation slide.
    """
    # Load test image
    test_img = cv2.imread(str(test_img_path))
    if test_img is None:
        return
    img_h, img_w = test_img.shape[:2]
    test_img_rgb = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
    ref_img_rgb = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)

    # 1. Extract Target Weights (The Signature)
    with torch.no_grad():
        predictor.set_image(ref_img)
        ref_features = predictor.get_image_embedding()
        mask_tensor = torch.from_numpy(ref_mask).float().unsqueeze(0).unsqueeze(0).to(ref_features.device)
        mask_resized = F.interpolate(mask_tensor, size=ref_features.shape[-2:], mode="bilinear")
        mask_resized = (mask_resized > 0.5).float()
        
        target_feat = ref_features * mask_resized
        target_weight = target_feat.sum(dim=(-1, -2)) / (mask_resized.sum(dim=(-1, -2)) + 1e-6)

    # 2. Generate Similarity Map on Test Image
    predictor.set_image(test_img)
    img_features = predictor.get_image_embedding()
    pooled_target_weight = target_weight.mean(dim=(-1, -2), keepdim=True)
    
    similarity_map = torch.mean(img_features * pooled_target_weight, dim=1, keepdim=True)
    similarity_map = F.interpolate(similarity_map, size=(img_h, img_w), mode="bilinear").squeeze().cpu().numpy()
    
    # Normalize similarity map for visualization
    sim_map_norm = (similarity_map - similarity_map.min()) / (similarity_map.max() - similarity_map.min() + 1e-8)

    # 3. Draw OBBs on Final Image
    final_img_drawn = test_img_rgb.copy()
    if Path(labels_path).exists():
        with open(labels_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = list(map(float, line.strip().split()))
                if len(parts) == 9: # class + 8 points for OBB
                    pts = np.array(parts[1:]).reshape(4, 2)
                    # Denormalize points
                    pts[:, 0] *= img_w
                    pts[:, 1] *= img_h
                    pts = pts.astype(np.int32)
                    cv2.polylines(final_img_drawn, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

    # 4. Create the Plot
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle("PerSAM DINOv2 Pipeline: Identity Targeting & OBB Extraction", fontsize=16, fontweight='bold')

    # Panel 1: Reference
    axes[0].imshow(ref_img_rgb)
    axes[0].imshow(ref_mask, alpha=0.5, cmap='jet')
    axes[0].set_title("1. One-Shot Reference")
    axes[0].axis('off')

    # Panel 2: Original Test Image
    axes[1].imshow(test_img_rgb)
    axes[1].set_title("2. Raw Frame")
    axes[1].axis('off')

    # Panel 3: Similarity Map
    im = axes[2].imshow(test_img_rgb)
    axes[2].imshow(sim_map_norm, cmap='jet', alpha=0.6) # Overlay heatmap
    axes[2].set_title("3. DINOv2 Similarity Map")
    axes[2].axis('off')

    # Panel 4: Final OBB output
    axes[3].imshow(final_img_drawn)
    axes[3].set_title("4. SAM + OBB Correction Head")
    axes[3].axis('off')

    plt.tight_layout()
    plt.savefig(output_vis_path, dpi=300)
    print(f"Saved visualization to: {output_vis_path}")
    plt.close()

# ---------------------------------------------------------
# Entry Point
# ---------------------------------------------------------
if __name__ == "__main__":
    # --- 1. Print the Metrics ---
    # Update these paths to match your actual directory structure
    TOTAL_PROCESSED_DIR = "datasets_refined/cowc/images"
    NEEDS_REFINEMENT_DIR = "datasets_refined/cowc/needs_refinement/images"
    
    calculate_manual_reduction(TOTAL_PROCESSED_DIR, NEEDS_REFINEMENT_DIR)

    # --- 2. Generate the Visuals ---
    print("Loading SAM for visualization...")
    # sam_predictor, _ = load_sam("models/sam_vit_b.pth")  # Use your existing load_sam func here
    
    # Placeholder for SAM loading (Assuming you have loaded it)
    sam = sam_model_registry["vit_b"](checkpoint="models/sam_vit_b.pth")
    sam.to("cuda" if torch.cuda.is_available() else "cpu")
    predictor = SamPredictor(sam)

    ref_image_path = "datasets/cowc/ref_car.png"
    ref_mask_path = "datasets/cowc/ref_mask.png"
    
    ref_img = cv2.imread(ref_image_path)
    ref_mask = cv2.imread(ref_mask_path, cv2.IMREAD_GRAYSCALE)
    
    if ref_img is not None and ref_mask is not None:
        # Pick one highly representative image from your dataset to show off
        test_img_file = "datasets/cowc_test_refined/cowc/images/10_12582.png"
        test_lbl_file = "datasets/cowc_test_refined/cowc/labels/10_12582.txt"
        out_vis_file  = "persam_pipeline_visualization.png"
        
        generate_visuals(predictor, ref_img, ref_mask, test_img_file, test_lbl_file, out_vis_file)
    else:
        print("[ERROR] Reference image or mask missing for visualization.")