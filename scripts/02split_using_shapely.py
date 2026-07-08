#!/usr/bin/env python3

import os
import cv2
import shutil
import numpy as np
from pathlib import Path
from shapely.geometry import Polygon, box

def split_yolo_obb_dataset(src_root, dst_root, tile_size=512, sub_size=256):
    src_root = Path(src_root)
    dst_root = Path(dst_root)
    
    if dst_root.exists():
        print(f"🧹 Clearing existing output directory: {dst_root}")
        shutil.rmtree(dst_root)
    dst_root.mkdir(parents=True, exist_ok=True)
    
    splits = ["train", "val", "test"]
    
    for split in splits:
        src_img_dir = src_root / "images" / split
        src_lbl_dir = src_root / "labels" / split
        
        if not src_img_dir.exists():
            continue
            
        print(f"✂️ Sharding [{split.upper()}] split from {tile_size}x{tile_size} to {sub_size}x{sub_size}...")
        
        dst_img_dir = dst_root / "images" / split
        dst_lbl_dir = dst_root / "labels" / split
        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_lbl_dir.mkdir(parents=True, exist_ok=True)
        
        img_files = sorted([f for f in os.listdir(src_img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        for img_name in img_files:
            stem = Path(img_name).stem
            img_path = src_img_dir / img_name
            lbl_path = src_lbl_dir / f"{stem}.txt"
            
            img = cv2.imread(str(img_path))
            if img is None:
                continue
                
            # Read 512x512 annotations
            labels = []
            if lbl_path.exists():
                with open(lbl_path, "r") as f:
                    for line in f.read().strip().splitlines():
                        parts = line.split()
                        if len(parts) == 9:
                            labels.append([int(parts[0])] + [float(x) for x in parts[1:]])
            
            # Divide 512x512 into four 256x256 blocks
            # (col, row) steps: (0,0), (256,0), (0,256), (256,256)
            for r in range(0, tile_size, sub_size):
                for c in range(0, tile_size, sub_size):
                    sub_img = img[r:r+sub_size, c:c+sub_size]
                    
                    # Target quadrant geometric box
                    quadrant_poly = box(c, r, c + sub_size, r + sub_size)
                    sub_labels = []
                    
                    for lbl in labels:
                        cls_id = lbl[0]
                        # Denormalize coordinates back to 512x512 space
                        pts = np.array(lbl[1:]).reshape(4, 2)
                        pts[:, 0] *= tile_size
                        pts[:, 1] *= tile_size
                        
                        car_poly = Polygon(pts)
                        
                        # Check if the car falls into this 256x256 quadrant
                        if car_poly.intersects(quadrant_poly):
                            intersection = car_poly.intersection(quadrant_poly)
                            
                            # Handle partial crops/slices gracefully
                            if intersection.geom_type == 'Polygon' and intersection.area > 5:
                                # Get the minimum area rotated bounding box of the remaining fragment
                                rect = intersection.minimum_rotated_rectangle
                                if rect.geom_type == 'Polygon':
                                    coords = list(rect.exterior.coords)[:4] # Grab 4 corners
                                    
                                    # Convert coordinates to the local 256x256 space and normalize
                                    norm_coords = []
                                    for x, y in coords:
                                        local_x = (x - c) / sub_size
                                        local_y = (y - r) / sub_size
                                        # Clip boundaries tightly
                                        norm_coords.append(max(0.0, min(1.0, local_x)))
                                        norm_coords.append(max(0.0, min(1.0, local_y)))
                                        
                                    sub_labels.append([cls_id] + norm_coords)
                                    
                    # Only save the image/label if it contains data or to keep splits balanced
                    sub_name = f"{stem}_q_{r}_{c}"
                    cv2.imwrite(str(dst_img_dir / f"{sub_name}.png"), sub_img)
                    
                    if sub_labels:
                        with open(dst_lbl_dir / f"{sub_name}.txt", "w") as out_f:
                            for sl in sub_labels:
                                coords_str = " ".join(f"{x:.6f}" for x in sl[1:])
                                out_f.write(f"{sl[0]} {coords_str}\n")
                                
    # Generate the fresh dataset config
    yaml_content = f"""path: {dst_root.absolute()}
train: images/train
val: images/val
test: images/test

nc: 1
names:
  0: vehicle
"""
    with open(dst_root / "dataset.yaml", "w") as y_f:
        y_f.write(yaml_content)
    print(f"🎉 Successfully generated a high-density 256x256 OBB dataset at: {dst_root}")

if __name__ == "__main__":
    # Point this to your completed 512x512 dataset directory
    SOURCE_DIR = "/Users/graddy/work/label-refiner/datasets/cowc_persam_refined"
    OUTPUT_DIR = "/Users/graddy/work/label-refiner/datasets/cowc_256_sharded"
    
    split_yolo_obb_dataset(SOURCE_DIR, OUTPUT_DIR)