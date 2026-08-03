#!/usr/bin/env python3
"""
Interactive viewer for inspecting DETR/DOTA training data. Loads images and their
corresponding label files, overlays standard or oriented bounding boxes using
absolute pixel coordinates, and provides a live zoom window that follows the mouse cursor.

Supports fast keyboard-based navigation:
- [A] / [Left Arrow] : Previous Image
- [D] / [Right Arrow]: Next Image
- [M]                : Move current image & label to 'needs_refinement' folder
- [Q] / [ESC]        : Quit
"""
import cv2
import os
import shutil
import numpy as np
import tkinter as tk
from tkinter import filedialog
from pathlib import Path

# --- CONFIGURATION ---
ZOOM_LEVEL = 8
CROP_SIZE = 80
WINDOW_NAME = "COWC DETR Inspector"
DISPLAY_HEIGHT = 800  # Adjust this to fit your screen

# TARGET SPLIT SELECTION: Change this to "train" or "val"
SPLIT = "train" 
# ---------------------

# Global state
mouse_raw = [0, 0]
scale_factor = 1.0

def mouse_callback(event, x, y, flags, param):
    global mouse_raw, scale_factor
    if event == cv2.EVENT_MOUSEMOVE:
        # Map window coordinates back to original image coordinates
        mouse_raw[0] = int(x / scale_factor)
        mouse_raw[1] = int(y / scale_factor)

def get_dataset_root():
    root = tk.Tk()
    root.withdraw()
    folder = filedialog.askdirectory(title="Select DETR/DOTA Root Dataset Directory (e.g. data/cowc_512_detr)")
    root.destroy()
    return folder

def draw_labels_detr(img, label_path, alpha=0.7):
    """
    Draws semi-transparent bounding boxes using DETR/DOTA absolute coordinates.
    alpha: 0.0 to 1.0 (transparency level)
    """
    if not os.path.exists(label_path): 
        return img

    # Create a transparent overlay layer
    overlay = img.copy()
    
    with open(label_path, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) < 8: continue
            
            # Color and Thickness
            color = (0, 255, 0) # Green for verified OBB targets
            
            try:
                # DETR format uses absolute integers directly
                pts = np.array([
                    [int(parts[0]), int(parts[1])],
                    [int(parts[2]), int(parts[3])],
                    [int(parts[4]), int(parts[5])],
                    [int(parts[6]), int(parts[7])]
                ], dtype=np.int32)
                
                # Draw true oriented polygon contour outline on overlay
                cv2.polylines(overlay, [pts.reshape((-1, 1, 2))], True, color, 1)
                
                # Optional: text tag right above the first corner coordinate 
                class_name = parts[8] if len(parts) > 8 else "vehicle"
                cv2.putText(overlay, class_name, (pts[0][0], max(12, pts[0][1] - 3)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            except ValueError:
                continue # Guard against text header contamination

    # Blend the overlay with the original image
    combined = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    return combined

def main():
    global scale_factor
    root_path = get_dataset_root()
    if not root_path: return

    # Paths configured dynamically via the SPLIT target selection variable
    root_dir = Path(root_path)
    img_dir = root_dir / "images" / SPLIT
    lbl_dir = root_dir / "labels" / SPLIT
    
    # Target directories for moved files (keeps splits clean inside needs_refinement)
    refine_img_dir = root_dir / "needs_refinement" / "images" / SPLIT
    refine_lbl_dir = root_dir / "needs_refinement" / "labels" / SPLIT
    
    if not img_dir.exists():
        print(f"Directory missing or unreadable: {img_dir}")
        print(f"Please verify your current layout matches the selected split: [{SPLIT}]")
        return

    # Updated pattern filters to support broad extensions
    img_files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff'))])
    
    if not img_files: 
        print(f"No images found inside target directory: {img_dir}")
        return

    idx = 0
    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

    print(f"🚀 DETR Inspector started. Mode: [{SPLIT.upper()}] split visualization pipeline.")

    while True:
        if not img_files:
            print("All images have been processed or moved. Exiting.")
            break

        img_name = img_files[idx]
        img_path = os.path.join(img_dir, img_name)
        lbl_name = img_name.rsplit('.', 1)[0] + ".txt"
        lbl_path = os.path.join(lbl_dir, lbl_name)
        
        raw_img = cv2.imread(img_path)
        if raw_img is None: 
            img_files.pop(idx)
            if img_files: idx = idx % len(img_files)
            continue
        
        # Swapped to our new DETR parsing loop
        labeled_img = draw_labels_detr(raw_img.copy(), lbl_path)
        h, w, _ = labeled_img.shape

        combined_w = w + h 
        scale_factor = DISPLAY_HEIGHT / h

        while True:
            # 1. Get Mouse in original image space
            mx, my = mouse_raw
            
            # 2. Create Zoom Pane
            zx, zy = np.clip(mx, 0, w), np.clip(my, 0, h)
            x1, y1 = max(0, zx - CROP_SIZE//2), max(0, zy - CROP_SIZE//2)
            x2, y2 = min(w, zx + CROP_SIZE//2), min(h, zy + CROP_SIZE//2)
            
            roi = labeled_img[y1:y2, x1:x2]
            zoom_pane = cv2.resize(roi, (h, h), interpolation=cv2.INTER_NEAREST)
            
            # ---------------------------------------------------------
            # Draw UI instructions and File Progress
            # ---------------------------------------------------------
            cv2.putText(zoom_pane, f"DETR SPLIT: {SPLIT.upper()}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(zoom_pane, "[M] Move to Refine", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # File/Progress Trackers
            cv2.putText(zoom_pane, f"File: {img_name}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(zoom_pane, f"Progress: {idx + 1} / {len(img_files)}", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # 3. Create Main Pane
            main_pane = labeled_img.copy()
            cv2.drawMarker(main_pane, (mx, my), (0, 0, 255), cv2.MARKER_CROSS, 30, 2)

            # 4. Concatenate and Resize for Display
            combined = np.hstack((main_pane, zoom_pane))
            final_w = int(combined_w * scale_factor)
            display_frame = cv2.resize(combined, (final_w, DISPLAY_HEIGHT))

            cv2.imshow(WINDOW_NAME, display_frame)
            
            key = cv2.waitKey(15) 
            if key != -1:
                k = key & 0xFF
                
                # Quit
                if k == ord('q') or k == 27: 
                    return
                
                # Next Image
                if k == ord('d') or key in [2555904, 83, 3]: 
                    idx = (idx + 1) % len(img_files)
                    break
                
                # Previous Image
                if k == ord('a') or key in [2424832, 81, 2]: 
                    idx = (idx - 1) % len(img_files)
                    break
                
                # Move for Refinement
                if k == ord('m'):
                    refine_img_dir.mkdir(parents=True, exist_ok=True)
                    refine_lbl_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Move image
                    shutil.move(img_path, refine_img_dir / img_name)
                    # Move label if it exists
                    if os.path.exists(lbl_path):
                        shutil.move(lbl_path, refine_lbl_dir / lbl_name)
                        
                    print(f"Moved [{SPLIT.upper()}]: {img_name} -> needs_refinement/../{SPLIT}/")
                    
                    # Remove from current queue and adjust index
                    img_files.pop(idx)
                    if img_files:
                        idx = idx % len(img_files) 
                    break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()