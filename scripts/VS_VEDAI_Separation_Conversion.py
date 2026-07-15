#!/usr/bin/env python3

import os
import shutil
from pathlib import Path

def process_vedai_dataset(src_img_dir, src_anno_dir, output_base_dir, map_to_vehicle_class=True):
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
    total_vehicles_dropped = 0
    
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
                # STRICT SEMANTIC FILTER:
                # Keep only Cars (1), Pickups (2), and Vans (9).
                # Drop Tractors, Campers, Boats, etc.
                # =========================================================
                if orig_class not in [1, 2, 9]:
                    total_vehicles_dropped += 1
                    continue
                    
                total_vehicles_kept += 1
                
                # Map to generic '0' if tracking as a single vehicle class
                class_id = 0 if map_to_vehicle_class else orig_class
                
                # VEDAI absolute image bounds for normalization (512x512 pixels)
                img_w, img_h = 512.0, 512.0
                
                # 1. Compute YOLO OBB Format (8 coordinate points)
                x_coords = [float(parts[6]), float(parts[7]), float(parts[8]), float(parts[9])]
                y_coords = [float(parts[10]), float(parts[11]), float(parts[12]), float(parts[13])]
                
                # Normalize between 0.0 and 1.0 and clip to be safe
                x_norm = [max(0.0, min(1.0, x / img_w)) for x in x_coords]
                y_norm = [max(0.0, min(1.0, y / img_h)) for y in y_coords]
                
                obb_lines.append(
                    f"{class_id} "
                    f"{x_norm[0]:.6f} {y_norm[0]:.6f} "
                    f"{x_norm[1]:.6f} {y_norm[1]:.6f} "
                    f"{x_norm[2]:.6f} {y_norm[2]:.6f} "
                    f"{x_norm[3]:.6f} {y_norm[3]:.6f}\n"
                )
                
                # =========================================================
                # CRITICAL FIX: Correct VEDAI Indexing
                # parts[0] is X-center, parts[1] is Y-center
                # =========================================================
                x_center = max(0.0, min(1.0, float(parts[0]) / img_w))
                y_center = max(0.0, min(1.0, float(parts[1]) / img_h))
                
                center_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} 0.005000 0.005000\n")
                
        # If no valid vehicles were found in this image after filtering, skip copying the image
        if len(obb_lines) == 0:
            continue
            
        # Save isolated outputs across all four operational directories
        
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
    print(f"  - Vehicles retained (Cars/Pickups/Vans): {total_vehicles_kept}")
    print(f"  - Outliers dropped (Tractors/Boats):     {total_vehicles_dropped}")
    print(f"Datasets generated under: {output_base_dir.resolve()}")

if __name__ == "__main__":
    # --- UPDATE THESE PATHS TO POINT TO YOUR EXTRACTED VEDAI FILE DIRECTORIES ---
    RAW_VEDAI_IMAGES = "data/VEDAI/Vehicules512" 
    RAW_VEDAI_ANNOTATIONS = "data/VEDAI/Annotations512"
    
    # Destination directory for the 4 separated test datasets
    OUTPUT_STUDY_DIR = "datasets/vedai"
    
    # Set to True to turn all retained vehicles into Class 0 
    MAP_TO_SINGLE_VEHICLE = True
    
    process_vedai_dataset(RAW_VEDAI_IMAGES, RAW_VEDAI_ANNOTATIONS, OUTPUT_STUDY_DIR, MAP_TO_SINGLE_VEHICLE)