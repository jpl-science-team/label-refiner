#!/usr/bin/env python3

"""
Interactive viewer for inspecting GroundingDINO detections.

Similar to your SAM dataset explorer, but loads ONLY the GDINO output
(your YOLO-format detection files) and overlays them live without saving
anything to disk.

Features:
    - Keyboard navigation (a=prev, d=next, q=quit)
    - Mouse-following zoom window
    - Supports standard YOLO boxes
    - Zero filesystem clutter (NO output saved)
"""

import cv2
import os
import numpy as np
import tkinter as tk
from tkinter import filedialog
from pathlib import Path

WINDOW_NAME = "GroundingDINO Explorer"
ZOOM_SIZE = 80
ZOOM_SCALE = 8
DISPLAY_HEIGHT = 800

mouse_pos = [0, 0]
scale_factor = 1.0


# -------------------------------------------------------------------------
# Mouse callback for zoom window
# -------------------------------------------------------------------------
def mouse_callback(event, x, y, flags, param):
    global mouse_pos, scale_factor
    if event == cv2.EVENT_MOUSEMOVE:
        mouse_pos[0] = int(x / scale_factor)
        mouse_pos[1] = int(y / scale_factor)


# -------------------------------------------------------------------------
# Folder picker
# -------------------------------------------------------------------------
def choose_dataset_folder():
    root = tk.Tk()
    root.withdraw()
    folder = filedialog.askdirectory(
        title="Select GDINO dataset root (contains images/ and labels/)"
    )
    root.destroy()
    return Path(folder) if folder else None


# -------------------------------------------------------------------------
# Draw YOLO detection boxes (GroundingDINO)
# -------------------------------------------------------------------------
def draw_gdino_boxes(img, lbl_path):
    h, w = img.shape[:2]
    if not lbl_path.exists():
        return img

    overlay = img.copy()

    with open(lbl_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            cls_id, cx, cy, bw, bh = map(float, parts)

            x1 = int((cx - bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w)
            y2 = int((cy + bh / 2) * h)

            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 200, 255), 2)
            cv2.putText(
                overlay,
                f"{int(cls_id)}",
                (x1, max(0, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 200, 255),
                2,
            )

    return overlay


# -------------------------------------------------------------------------
# Main interactive viewer
# -------------------------------------------------------------------------
def main():
    global scale_factor

    root = choose_dataset_folder()
    if root is None:
        print("No dataset selected.")
        return

    img_dir = root / "images"
    lbl_dir = root / "labels"

    if not img_dir.exists() or not lbl_dir.exists():
        print("Selected folder must contain /images and /labels.")
        return

    img_files = sorted([
        f for f in os.listdir(img_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])

    if not img_files:
        print("No images found in dataset.")
        return

    idx = 0

    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

    while True:
        img_path = img_dir / img_files[idx]
        lbl_path = lbl_dir / f"{Path(img_files[idx]).stem}.txt"

        img = cv2.imread(str(img_path))
        if img is None:
            idx = (idx + 1) % len(img_files)
            continue

        # Draw GDINO detections
        display = draw_gdino_boxes(img.copy(), lbl_path)
        h, w = display.shape[:2]

        # scale image to fixed display height
        scale_factor = DISPLAY_HEIGHT / h
        disp_w = int(w * scale_factor)
        resized = cv2.resize(display, (disp_w, DISPLAY_HEIGHT))

        # local coordinates for zoom
        mx, my = mouse_pos
        mx = int(np.clip(mx, 0, w - 1))
        my = int(np.clip(my, 0, h - 1))

        # zoom region from full-res before resizing
        x1 = max(mx - ZOOM_SIZE // 2, 0)
        y1 = max(my - ZOOM_SIZE // 2, 0)
        x2 = min(mx + ZOOM_SIZE // 2, w)
        y2 = min(my + ZOOM_SIZE // 2, h)
        roi = display[y1:y2, x1:x2]

        zoom = cv2.resize(
            roi,
            (ZOOM_SIZE * ZOOM_SCALE, ZOOM_SIZE * ZOOM_SCALE),
            interpolation=cv2.INTER_NEAREST
        )
        cv2.putText(zoom, "ZOOM", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # final combined output (image + zoom)
        combined = np.hstack([resized, zoom])

        cv2.imshow(WINDOW_NAME, combined)

        # ---------------------------------------------------------
        # Keyboard navigation
        # ---------------------------------------------------------
        key = cv2.waitKey(10) & 0xFF

        if key in (ord("q"), 27):
            break
        elif key in (ord("d"), 83, 3):   # next → key or right-arrow
            idx = (idx + 1) % len(img_files)
        elif key in (ord("a"), 81, 2):   # prev ← key or left-arrow
            idx = (idx - 1) % len(img_files)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()