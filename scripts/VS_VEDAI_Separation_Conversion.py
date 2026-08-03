#!/usr/bin/env python3

import os
import shutil
import cv2
import numpy as np
from pathlib import Path

def process_vedai_dataset(src_img_dir, src_anno_dir, output_base_dir, map_to_vehicle_class=True, apply_strict_geom_filter=False):
    src_img_dir = Path(src_img_dir)
    src_anno_dir = Path(src_anno_dir)
    output_base_dir = Path(output_base_dir)
    
    # Define the 4 target dataset paths
    dirs = {
        "co_bbox": output_base_dir / "vedai_color_obb",
        "ir_bbox": output_base_dir / "vedai_ir_obb",
        "co_center": output_base_dir / "vedai_color_centerpoint",
        "ir_center": output_base_dir / "vedai_ir_centerpoint"
    }
    
    # Create directory trees
    for d_path in dirs.values():
        (d_path / "images").mkdir(parents=True, exist_ok=True)
        (d_path / "labels").mkdir(parents=True, exist_ok=True)
        
    # Find all native annotation text files
    anno_files = list(src_anno_dir.glob("*.txt"))
    print(f"Found {len(anno_files)} annotation files to process.")
    
    success_count = 0
    total_vehicles_kept = 0
    total_vehicles_dropped_semantic = 0
    total_vehicles_dropped_geometric = 0
    
    # 1024x1024 dimensions
    img_w, img_h = 1024.0, 1024.0
    
    for anno_path in anno_files:
        base_name = anno_path.stem  # e.g., '00000001'
        
        # Look for both corresponding images (handles .png and .jpg variants)
        co_img_path = None
        ir_img_path = None
        for ext in ['.png', '.jpg', '.jpeg']:
            if (src_img_dir / f"{base_name}_co{ext}").exists():
                co_img_path = src_img_dir / f"{base_name}_co{ext}"
            if (src_img_dir / f"{base_name}_ir{ext}").exists():
                ir_img_path = src_img_dir / f"{base_name}_ir{ext}"
                
        # Skip processing if images are missing for this annotation block
        if not co_img_path or not ir_img_path:
            continue
            
        obb_lines = []
        center_lines = []
        
        with open(anno_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 14:
                    continue
                
                orig_class = int(parts[3])
                
                # =========================================================
                # 1. SEMANTIC FILTER
                # Standard VEDAI Road Vehicles: 1=Car, 2=Pickup, 4=Truck, 9=Van
                # Add 5=Semi, 7=Tractor, 8=Camping Car if you want all ground vehicles.
                # =========================================================
                road_vehicles = {1, 2, 4, 5, 7, 8, 9}
                if orig_class not in road_vehicles:
                    total_vehicles_dropped_semantic += 1
                    continue
                    
                # Extract original pixel coordinates
                x_coords = [float(parts[6]), float(parts[7]), float(parts[8]), float(parts[9])]
                y_coords = [float(parts[10]), float(parts[11]), float(parts[12]), float(parts[13])]
                
                # =========================================================
                # 2. OPTIONAL GEOMETRIC FILTER
                # Default is FALSE: We keep all annotated ground targets
                # =========================================================
                if apply_strict_geom_filter:
                    pts = np.array([
                        [x_coords[0], y_coords[0]],
                        [x_coords[1], y_coords[1]],
                        [x_coords[2], y_coords[2]],
                        [x_coords[3], y_coords[3]]
                    ], dtype=np.float32)
                    
                    rect = cv2.minAreaRect(pts)
                    (cx, cy), (w, h), angle = rect
                    max_dim = max(w, h)
                    min_dim = min(w, h)
                    aspect = max_dim / (min_dim + 1e-6)
                    
                    is_bad_box = max_dim > 95 or max_dim < 12 or aspect > 3.5
                    if is_bad_box:
                        total_vehicles_dropped_geometric += 1
                        continue

                # Box retained!
                total_vehicles_kept += 1
                class_id = 0 if map_to_vehicle_class else orig_class
                
                # Compute YOLO OBB Format (8 coordinate points normalized)
                x_norm = [max(0.0, min(1.0, x / img_w)) for x in x_coords]
                y_norm = [max(0.0, min(1.0, y / img_h)) for y in y_coords]
                
                obb_lines.append(
                    f"{class_id} "
                    f"{x_norm[0]:.6f} {y_norm[0]:.6f} "
                    f"{x_norm[1]:.6f} {y_norm[1]:.6f} "
                    f"{x_norm[2]:.6f} {y_norm[2]:.6f} "
                    f"{x_norm[3]:.6f} {y_norm[3]:.6f}\n"
                )
                
                # Centerpoints Format (x_center, y_center, dummy_w, dummy_h)
                x_center = max(0.0, min(1.0, float(parts[0]) / img_w))
                y_center = max(0.0, min(1.0, float(parts[1]) / img_h))
                center_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} 0.005000 0.005000\n")
                
        # Save output image-annotation pairs
        # Color (CO) Outputs
        shutil.copy(co_img_path, dirs["co_bbox"] / "images" / co_img_path.name)
        shutil.copy(co_img_path, dirs["co_center"] / "images" / co_img_path.name)
        
        with open(dirs["co_bbox"] / "labels" / f"{co_img_path.stem}.txt", "w") as f:
            f.writelines(obb_lines)
        with open(dirs["co_center"] / "labels" / f"{co_img_path.stem}.txt", "w") as f:
            f.writelines(center_lines)
            
        # Infrared (IR) Outputs
        shutil.copy(ir_img_path, dirs["ir_bbox"] / "images" / ir_img_path.name)
        shutil.copy(ir_img_path, dirs["ir_center"] / "images" / ir_img_path.name)
        
        with open(dirs["ir_bbox"] / "labels" / f"{ir_img_path.stem}.txt", "w") as f:
            f.writelines(obb_lines)
        with open(dirs["ir_center"] / "labels" / f"{ir_img_path.stem}.txt", "w") as f:
            f.writelines(center_lines)
            
        success_count += 1

    print(f"\nProcessing complete! Successfully structured {success_count} aligned image pairs.")
    print(f"  - Retained Vehicle Targets: {total_vehicles_kept}")
    print(f"  - Dropped by Semantic Class: {total_vehicles_dropped_semantic}")
    print(f"  - Dropped by Geometric Filter: {total_vehicles_dropped_geometric}")
    print(f"Datasets generated under: {output_base_dir.resolve()}")

if __name__ == "__main__":
    RAW_VEDAI_IMAGES = "data/VEDAI/Vehicules1024" 
    RAW_VEDAI_ANNOTATIONS = "data/VEDAI/Annotations1024"
    OUTPUT_STUDY_DIR = "datasets/vedai_1024"
    MAP_TO_SINGLE_VEHICLE = True
    
    # Set apply_strict_geom_filter=False to keep all valid bounding boxes
    process_vedai_dataset(
        RAW_VEDAI_IMAGES, 
        RAW_VEDAI_ANNOTATIONS, 
        OUTPUT_STUDY_DIR, 
        map_to_vehicle_class=MAP_TO_SINGLE_VEHICLE,
        apply_strict_geom_filter=False
    )