#!/usr/bin/env python3

"""
Optimized GroundingDINO vehicle detection for Multi-GPU (Tesla M60s).

Features:
    • Multi-GPU Parallelism (maps workers dynamically to available GPUs)
    • Hardware-aware: FP16 is explicitly disabled (Maxwell M60 GPUs lack native FP16; FP32 is faster)
    • Aggressive tiling (crop=256)
    • Clean, self-contained, no CLI args
    • Writes only YOLO boxes, no extra output
"""

import os
import cv2
import torch
import numpy as np
import multiprocessing as mp
from pathlib import Path

from groundingdino.util.inference import Model

# -----------------------------------------------------------
# User-edited VARIABLES
# -----------------------------------------------------------
CROP_SIZE = 256
TEXT_PROMPT = "vehicle"
OUTPUT_THRESHOLD = 0.25

# -----------------------------------------------------------
# Worker Globals (Initialized per process)
# -----------------------------------------------------------
worker_model = None
worker_device = None

def init_worker(gpu_queue):
    """
    Initializes a GroundingDINO model on a specific GPU for each worker process.
    """
    global worker_model, worker_device

    # Grab an available GPU ID from the queue
    gpu_id = gpu_queue.get()
    worker_device = f"cuda:{gpu_id}"

    print(f"[INFO] Initializing worker on {worker_device} (FP32 Mode)...")
    
    # Load model strictly in FP32 (Best for Tesla M60 / Maxwell)
    worker_model = Model(
        model_config_path="external/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        model_checkpoint_path="models/groundingdino/groundingdino_swint_ogc.pth",
        device=worker_device,
    )

def run_dino_on_crop(crop_img):
    """Returns list of YOLO-format detections (class, cx, cy, w, h)."""
    global worker_model

    # Inference mode skips gradients, speeding up execution
    with torch.inference_mode():
        detections = worker_model.predict_with_classes(
            image=crop_img,
            classes=[TEXT_PROMPT],
            box_threshold=OUTPUT_THRESHOLD,
            text_threshold=OUTPUT_THRESHOLD,
        )

    yolo_boxes = []
    h, w = crop_img.shape[:2]

    for d in detections:
        x1, y1, x2, y2 = d["box"]
        cx = (x1 + (x2 - x1) / 2) / w
        cy = (y1 + (y2 - y1) / 2) / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h
        yolo_boxes.append((0, cx, cy, bw, bh))  # class 0 fixed

    return yolo_boxes

def crop_to_full_image(box, crop_x, crop_y, crop_w, crop_h, img_w, img_h):
    """Converts from crop coordinates back to full image coordinates."""
    cls_id, cx, cy, bw, bh = box
    abs_cx = crop_x + cx * crop_w
    abs_cy = crop_y + cy * crop_h
    abs_w = bw * crop_w
    abs_h = bh * crop_h

    return (cls_id, abs_cx / img_w, abs_cy / img_h, abs_w / img_w, abs_h / img_h)

def process_image(args):
    """Worker function to process a single image and save its labels."""
    img_path, lbl_path, out_img_dir, out_lbl_dir, img_name = args

    img = cv2.imread(str(img_path))
    if img is None:
        return

    img_h, img_w = img.shape[:2]
    cv2.imwrite(str(out_img_dir / img_name), img)

    with open(lbl_path, "r") as f:
        labels = f.read().strip().splitlines()

    all_detections = []

    for line in labels:
        parts = list(map(float, line.split()))
        if len(parts) < 5:
            continue

        _, cx, cy, w, h = parts

        px = int(cx * img_w)
        py = int(cy * img_h)

        x1 = max(0, px - CROP_SIZE // 2)
        y1 = max(0, py - CROP_SIZE // 2)
        x2 = min(img_w, x1 + CROP_SIZE)
        y2 = min(img_h, y1 + CROP_SIZE)

        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        crop_boxes = run_dino_on_crop(crop)

        for b in crop_boxes:
            full_box = crop_to_full_image(b, x1, y1, (x2 - x1), (y2 - y1), img_w, img_h)
            all_detections.append(full_box)

    # Write YOLO output
    out_lbl_path = out_lbl_dir / (Path(img_name).stem + ".txt")
    with open(out_lbl_path, "w") as f:
        for cls_id, cx, cy, bw, bh in all_detections:
            f.write(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")


# -----------------------------------------------------------
# Task Distributor and Entry Point
# -----------------------------------------------------------
def run_grounding_dino_parallel(input_root, output_root):
    datasets = ["cowc", "cowc_rgb_1m", "cowc_rgb_05m"]
    input_root = Path(input_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    # 1. Collect all valid image processing tasks
    tasks = []
    for dataset in datasets:
        in_img_dir = input_root / dataset / "images"
        in_lbl_dir = input_root / dataset / "labels"
        out_img_dir = output_root / dataset / "images"
        out_lbl_dir = output_root / dataset / "labels"

        if not in_img_dir.exists():
            print(f"[WARN] Missing {in_img_dir}, skipping.")
            continue

        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_lbl_dir.mkdir(parents=True, exist_ok=True)

        image_list = sorted(
            f for f in os.listdir(in_img_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))
        )

        for img_name in image_list:
            img_path = in_img_dir / img_name
            lbl_path = in_lbl_dir / (Path(img_name).stem + ".txt")

            if lbl_path.exists():
                tasks.append((img_path, lbl_path, out_img_dir, out_lbl_dir, img_name))

    if not tasks:
        print("[INFO] No tasks to process.")
        return

    # 2. Setup Multi-GPU Queue
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No GPUs detected. CUDA is required.")

    print(f"\n[INFO] Found {num_gpus} GPUs. Setting up worker pool...")
    
    manager = mp.Manager()
    gpu_queue = manager.Queue()
    
    # Assign one worker process per available GPU (ideal for 8GB VRAM limits)
    for i in range(num_gpus):
        gpu_queue.put(i)

    # 3. Process images in parallel
    # `imap_unordered` yields results as soon as they are ready, regardless of task order
    with mp.Pool(processes=num_gpus, initializer=init_worker, initargs=(gpu_queue,)) as pool:
        for i, _ in enumerate(pool.imap_unordered(process_image, tasks), 1):
            if i % 10 == 0 or i == len(tasks):
                print(f"[PROGRESS] Processed {i}/{len(tasks)} images...")

    print("\n✔ Finished processing all datasets.")


if __name__ == "__main__":
    # REQUIRED: 'spawn' is necessary for PyTorch to initialize CUDA in subprocesses safely
    mp.set_start_method('spawn', force=True)
    
    run_grounding_dino_parallel(
        input_root="datasets/cowc_test_subset",
        output_root="datasets/cowc_test_gdino"
    )