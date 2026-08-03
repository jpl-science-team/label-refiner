import os
import shutil

def clone_and_convert_dataset(src_root, dest_root, img_size=512):
    """
    1. Copies images from src_root to dest_root into proper MMRotate split layouts.
    2. Converts YOLO-OBB normalized labels into absolute DOTA/DETR labels.
    """
    splits = ['train', 'val']
    
    for split in splits:
        print(f"--- Processing Split: {split} ---")
        
        # Define source paths (YOLO format)
        src_img_dir = os.path.join(src_root, 'images', split)
        src_lbl_dir = os.path.join(src_root, 'labels', split)
        
        # Define destination paths (Strict MMRotate/DOTA format: split/images and split/annfiles)
        dest_img_dir = os.path.join(dest_root, split, 'images')
        dest_lbl_dir = os.path.join(dest_root, split, 'annfiles')
        
        # Create directories cleanly
        os.makedirs(dest_img_dir, exist_ok=True)
        os.makedirs(dest_lbl_dir, exist_ok=True)
        
        # --- STEP 1: Copy Images ---
        if os.path.exists(src_img_dir):
            img_files = [f for f in os.listdir(src_img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
            print(f"Copying {len(img_files)} images to {dest_img_dir}...")
            for img in img_files:
                shutil.copy2(os.path.join(src_img_dir, img), os.path.join(dest_img_dir, img))
        else:
            print(f"Warning: Source image directory missing: {src_img_dir}")

        # --- STEP 2: Convert and Copy Labels ---
        if os.path.exists(src_lbl_dir):
            lbl_files = [f for f in os.listdir(src_lbl_dir) if f.endswith('.txt')]
            print(f"Converting and saving {len(lbl_files)} labels to {dest_lbl_dir}...")
            
            for filename in lbl_files:
                with open(os.path.join(src_lbl_dir, filename), "r") as f:
                    lines = f.readlines()
                    
                dota_lines = []
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) < 9:
                        continue  # Skip any malformed annotations
                        
                    # Extract coords (bypassing class index)
                    coords = [float(x) for x in parts[1:]]
                    
                    # Scale relative coordinates (0.0 - 1.0) back up to absolute pixels
                    x1, y1 = int(coords[0] * img_size), int(coords[1] * img_size)
                    x2, y2 = int(coords[2] * img_size), int(coords[3] * img_size)
                    x3, y3 = int(coords[4] * img_size), int(coords[5] * img_size)
                    x4, y4 = int(coords[6] * img_size), int(coords[7] * img_size)
                    
                    # DOTA standard layout: x1 y1 x2 y2 x3 y3 x4 y4 class_name difficulty
                    dota_lines.append(f"{x1} {y1} {x2} {y2} {x3} {y3} {x4} {y4} vehicle 0\n")
                    
                with open(os.path.join(dest_lbl_dir, filename), "w") as f:
                    f.writelines(dota_lines)
        else:
            print(f"Warning: Source label directory missing: {src_lbl_dir}")

    print("\n[SUCCESS] Dedicated DETR dataset created cleanly at:", dest_root)

if __name__ == "__main__":
    clone_and_convert_dataset(
        src_root="datasets/cowc_512_sensor_degraded",
        dest_root="datasets/cowc_512_degraded_dota",
        img_size=512
    )