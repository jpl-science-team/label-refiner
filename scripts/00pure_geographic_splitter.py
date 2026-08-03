'''
Takes the raw COWC dataset and splits the dataset to create Train, Val, and Test splits. Extracts the annoations and saves then in YOLO format. 
'''
import os
import glob
import cv2
import numpy as np
from tqdm import tqdm

# --- Configuration ---
DATA_DIR = "data/COWC"  
OUTPUT_DIR = "datasets/20260727_COWC_Points_512"
WINDOW_SIZE = 512 # How big you want the tiles 1024, 512, 256, 128 etc.
STRIDE = 256  # 50% overlap guarantees cars are fully in at least one frame

# Approximate bounding box size in pixels to encapsulate a car around the dot
CAR_SIZE = 32  
CLASS_ID = 0   # 0 for car

# Splits by City (Test split dissolved into train/val)
SPLITS = {
    "train": ["Toronto_ISPRS", "Selwyn_LINZ", "Utah_AGRC"],
    "val": ["Vaihingen_ISPRS"],
    "test": ["Potsdam_ISPRS", "Columbus_CSUAV_AFRL"]
}

def setup_directories():
    for split in SPLITS.keys():
        os.makedirs(os.path.join(OUTPUT_DIR, split, "images"), exist_ok=True)
        # Standardized to 'labels' directory for YOLO
        os.makedirs(os.path.join(OUTPUT_DIR, split, "labels"), exist_ok=True)

def process_city_images(city_dir):
    """ Finds matching image and annotation pairs in a city folder """
    all_pngs = glob.glob(os.path.join(DATA_DIR, city_dir, "*.png"))
    pairs = []
    
    for img_path in all_pngs:
        if "_Annotated_" in img_path:
            continue
        
        base, ext = os.path.splitext(img_path)
        anno_path = f"{base}_Annotated_Cars.png"
        
        if os.path.exists(anno_path):
            pairs.append((img_path, anno_path))
            
    return pairs

def extract_patches(img_path, anno_path, split_name, city_name, global_count):
    img = cv2.imread(img_path)
    if img is None:
        return global_count
        
    anno = cv2.imread(anno_path)
    anno_gray = cv2.cvtColor(anno, cv2.COLOR_BGR2GRAY)
    _, pts_mask = cv2.threshold(anno_gray, 1, 255, cv2.THRESH_BINARY)
    
    h, w, _ = img.shape
    base_filename = os.path.splitext(os.path.basename(img_path))[0]
    
    # Constant normalized width/height dimensions for YOLO bounding boxes
    norm_w = CAR_SIZE / WINDOW_SIZE
    norm_h = CAR_SIZE / WINDOW_SIZE
    
    # Sliding window
    for y in range(0, h - WINDOW_SIZE + 1, STRIDE):
        for x in range(0, w - WINDOW_SIZE + 1, STRIDE):
            
            patch_img = img[y:y+WINDOW_SIZE, x:x+WINDOW_SIZE]
            patch_mask = pts_mask[y:y+WINDOW_SIZE, x:x+WINDOW_SIZE]
            
            y_indices, x_indices = np.where(patch_mask > 0)
            
            # Skip empty patches to keep training focused
            if len(x_indices) == 0:
                continue
                
            label_file_name = f"{city_name}_{base_filename}_patch_{y}_{x}.txt"
            img_file_name = f"{city_name}_{base_filename}_patch_{y}_{x}.png"
            
            # Paths adjusted to target standard YOLO layout
            label_out_path = os.path.join(OUTPUT_DIR, split_name, "labels", label_file_name)
            img_out_path = os.path.join(OUTPUT_DIR, split_name, "images", img_file_name)
            
            with open(label_out_path, "w") as f:
                for px, py in zip(x_indices, y_indices):
                    # Convert absolute patch pixel locations to normalized YOLO format
                    norm_x_center = px / WINDOW_SIZE
                    norm_y_center = py / WINDOW_SIZE
                    
                    # Write in standard space-separated YOLO layout
                    f.write(f"{CLASS_ID} {norm_x_center:.6f} {norm_y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n")
            
            cv2.imwrite(img_out_path, patch_img)
            global_count += 1
            
    return global_count

def main():
    setup_directories()
    print("Starting YOLO formatted COWC preprocessing pipeline...\n")
    
    for split, cities in SPLITS.items():
        print(f"--- Processing Split: {split.upper()} ---")
        split_counter = 0
        
        for city in cities:
            if not os.path.exists(os.path.join(DATA_DIR, city)):
                print(f"Warning: Directory {city} not found. Skipping.")
                continue
                
            pairs = process_city_images(city)
            print(f"  Found {len(pairs)} large scale images in {city}")
            
            for img_path, anno_path in tqdm(pairs, desc=f"  Extracting {city}"):
                split_counter = extract_patches(img_path, anno_path, split, city, split_counter)
                
        print(f"Generated {split_counter} patches for {split}.\n")

if __name__ == "__main__":
    main()