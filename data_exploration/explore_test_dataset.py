"""
Interactive viewer for inspecting refined training data. Loads images and their
corresponding label files, overlays standard or oriented bounding boxes, and
provides a live zoom window that follows the mouse cursor. Supports fast
keyboard-based navigation for reviewing annotation quality across a dataset.
"""
import cv2
import os
import numpy as np
import tkinter as tk
from tkinter import filedialog
from pathlib import Path

WINDOW_NAME = "Dataset Explorer"
ZOOM_SIZE = 80
ZOOM_SCALE = 8
DISPLAY_HEIGHT = 800

mouse_pos = [0, 0]
scale_factor = 1.0

def mouse_callback(event, x, y, flags, param):
    global mouse_pos, scale_factor
    if event == cv2.EVENT_MOUSEMOVE:
        mouse_pos[0] = int(x / scale_factor)
        mouse_pos[1] = int(y / scale_factor)

def choose_dataset_folder():
    root = tk.Tk()
    root.withdraw()
    folder = filedialog.askdirectory(title="Select dataset folder containing images & labels")
    root.destroy()
    return Path(folder) if folder else None

def draw_labels(img, lbl_path):
    h, w = img.shape[:2]
    if not lbl_path.exists():
        return img

    overlay = img.copy()

    with open(lbl_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            vals = list(map(float, parts[1:]))

            # Polygon / oriented YOLO
            if len(vals) == 8:
                pts = np.array([[vals[i]*w, vals[i+1]*h] for i in range(0,8,2)], np.int32)
                cv2.polylines(overlay, [pts.reshape((-1,1,2))], True, (0,255,255), 2)

            # Standard YOLO
            elif len(vals) == 4:
                cx, cy, bw, bh = vals
                x1 = int((cx - bw/2) * w)
                y1 = int((cy - bh/2) * h)
                x2 = int((cx + bw/2) * w)
                y2 = int((cy + bh/2) * h)
                cv2.rectangle(overlay, (x1,y1), (x2,y2), (0,255,0), 2)

    return overlay

def main():
    global scale_factor

    root = choose_dataset_folder()
    if root is None:
        print("No dataset chosen.")
        return

    img_dir = root / "images"
    lbl_dir = root / "labels"

    if not img_dir.exists() or not lbl_dir.exists():
        print("Dataset must contain /images and /labels folders.")
        return

    img_files = sorted([f for f in os.listdir(img_dir)
                        if f.lower().endswith(('.png','.jpg','.jpeg'))])

    if not img_files:
        print("No images found.")
        return

    idx = 0
    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

    while True:
        img_path = img_dir / img_files[idx]
        lbl_path = lbl_dir / (img_files[idx].rsplit('.', 1)[0] + ".txt")

        img = cv2.imread(str(img_path))
        if img is None:
            idx = (idx + 1) % len(img_files)
            continue

        labeled = draw_labels(img.copy(), lbl_path)
        h, w = labeled.shape[:2]

        scale_factor = DISPLAY_HEIGHT / h

        # cursor pos in original coords
        mx, my = mouse_pos
        mx = int(np.clip(mx, 0, w - 1))
        my = int(np.clip(my, 0, h - 1))

        # zoom crop
        x1 = max(mx - ZOOM_SIZE//2, 0)
        y1 = max(my - ZOOM_SIZE//2, 0)
        x2 = min(mx + ZOOM_SIZE//2, w)
        y2 = min(my + ZOOM_SIZE//2, h)
        roi = labeled[y1:y2, x1:x2]

        zoom = cv2.resize(roi, (h, h), interpolation=cv2.INTER_NEAREST)
        cv2.putText(zoom, "ZOOM", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

        combined = np.hstack([labeled, zoom])
        disp_w = int(combined.shape[1] * scale_factor)
        frame = cv2.resize(combined, (disp_w, DISPLAY_HEIGHT))

        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(10) & 0xFF

        if key in (ord('q'), 27):
            break
        elif key in (ord('d'), 83, 3):   # right / next
            idx = (idx + 1) % len(img_files)
        elif key in (ord('a'), 81, 2):   # left / prev
            idx = (idx - 1) % len(img_files)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

