#!/usr/bin/env python3

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from shapely.geometry import Polygon

def obb_to_polygon(obb_coords, img_w=1024.0, img_h=1024.0):
    """Converts 8 normalized OBB coordinates [x1, y1, x2, y2, x3, y3, x4, y4] to a Shapely Polygon."""
    pts = np.array(obb_coords).reshape(4, 2)
    # Scale back to absolute pixel dimensions for exact polygon math
    pts[:, 0] *= img_w
    pts[:, 1] *= img_h
    poly = Polygon(pts)
    if not poly.is_valid:
        poly = poly.buffer(0)
    return poly

def compute_obb_iou(poly1, poly2):
    """Computes the exact Intersection over Union between two oriented polygons."""
    try:
        inter_area = poly1.intersection(poly2).area
        union_area = poly1.area + poly2.area - inter_area
        return inter_area / (union_area + 1e-6)
    except Exception:
        return 0.0

def load_yolo_obb_file(file_path):
    """Parses a YOLO OBB text file returning a list of dicts: {'class': int, 'poly': Polygon}."""
    targets = []
    if not file_path.exists():
        return targets
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 9:
                continue
            cls_id = int(parts[0])
            coords = [float(x) for x in parts[1:9]]
            poly = obb_to_polygon(coords)
            targets.append({'class': cls_id, 'poly': poly})
    return targets

def run_evaluation(gt_dir, pred_dir, output_dir, iou_threshold=0.50):
    gt_root = Path(gt_dir)
    pred_root = Path(pred_dir)
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    gt_lbl_dir = gt_root / "labels" if (gt_root / "labels").exists() else gt_root
    pred_lbl_dir = pred_root / "labels" if (pred_root / "labels").exists() else pred_root

    gt_files = list(gt_lbl_dir.glob("*.txt"))
    print(f"🧐 Found {len(gt_files)} ground-truth text annotations to validate...")

    # We account for vehicle (0) and background/missed (1) for confusion matrix structure
    # Class 0: Vehicle, Class 1: Background (used to track FPs / FNs cleanly)
    confusion_data = {"TP": 0, "FP": 0, "FN": 0}
    all_ious = []

    for gt_path in gt_files:
        pred_path = pred_lbl_dir / gt_path.name
        
        gt_boxes = load_yolo_obb_file(gt_path)
        pred_boxes = load_yolo_obb_file(pred_path)

        matched_preds = set()

        for gt_box in gt_boxes:
            best_iou = 0.0
            best_pred_idx = -1
            
            for idx, pred_box in enumerate(pred_boxes):
                if idx in matched_preds:
                    continue
                iou = compute_obb_iou(gt_box['poly'], pred_box['poly'])
                if iou > best_iou:
                    best_iou = iou
                    best_pred_idx = idx

            if best_iou >= iou_threshold:
                confusion_data["TP"] += 1
                matched_preds.add(best_pred_idx)
                all_ious.append(best_iou)
            else:
                confusion_data["FN"] += 1  # Human labeled it, pipeline missed it

        # Any predicted box that didn't match a human label is an auto-labeling False Positive
        fps_in_file = len(pred_boxes) - len(matched_preds)
        confusion_data["FP"] += fps_in_file

    # Compute Core Machine Learning Pipeline Metrics
    tp, fp, fn = confusion_data["TP"], confusion_data["FP"], confusion_data["FN"]
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-6)
    mean_iou = np.mean(all_ious) if all_ious else 0.0

    # Write out diagnostic metrics report text file
    report_path = out_root / "pipeline_evaluation_report.txt"
    with open(report_path, "w") as f:
        f.write("==================================================\n")
        f.write("   DINOv2 + PerSAM AUTO-LABELING VALIDATION REPORT\n")
        f.write("==================================================\n")
        f.write(f"Ground Truth Directory : {gt_lbl_dir.resolve()}\n")
        f.write(f"Pipeline Prediction Dir: {pred_lbl_dir.resolve()}\n")
        f.write(f"IoU Matching Threshold : {iou_threshold}\n")
        f.write("--------------------------------------------------\n")
        f.write(f"True Positives (TP)    : {tp}\n")
        f.write(f"False Positives (FP)   : {fp}  <-- (Shadows/Pavement Errors)\n")
        f.write(f"False Negatives (FN)   : {fn}  <-- (Missed Vehicles)\n")
        f.write("--------------------------------------------------\n")
        f.write(f"Precision              : {precision:.4f}\n")
        f.write(f"Recall                 : {recall:.4f}\n")
        f.write(f"F1-Score               : {f1_score:.4f}\n")
        f.write(f"Mean Oriented IoU (mIoU): {mean_iou:.4f}\n")
        f.write("==================================================\n")

    print(f"\n📊 Performance Metrics Logged to: {report_path}")

    # =======================================================
    # 📉 GENERATING GRAPHS FOR YOUR PRESENTATION
    # =======================================================
    sns.set_theme(style="whitegrid")
    
    # Graph 1: 2x2 Normalized OBB Confusion Matrix
    cm = np.array([[tp, fn], 
                   [fp, 0]]) # standard object detection evaluation layout (no true negative background tracking)
    
    plt.figure(figsize=(7, 5))
    labels = ['Vehicle', 'Background\n(Missed/Shadow)']
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, cbar=False, annot_kws={"size": 14})
    plt.ylabel('Human Ground Truth', fontsize=12, fontweight='bold')
    plt.xlabel('PerSAM+DINOv2 Pipeline', fontsize=12, fontweight='bold')
    plt.title('Oriented Bounding Box Confusion Matrix', fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(out_root / "confusion_matrix.png", dpi=300)
    plt.close()

    # Graph 2: Summary Metrics Bar Chart
    plt.figure(figsize=(8, 4.5))
    metrics_elements = ['Precision', 'Recall', 'F1-Score', 'Mean IoU']
    values_elements = [precision, recall, f1_score, mean_iou]
    colors = ['#1f77b4', '#aec7e8', '#ff7f0e', '#2ca02c']
    
    bars = plt.bar(metrics_elements, values_elements, color=colors, width=0.5, edgecolor='black', linewidth=0.7)
    plt.ylim(0, 1.05)
    plt.title('PerSAM + DINOv2 Pipeline Summary Metrics', fontsize=14, fontweight='bold', pad=15)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f"{yval:.3f}", ha='center', va='bottom', fontsize=11, fontweight='bold')
        
    plt.tight_layout()
    plt.savefig(out_root / "pipeline_summary_metrics.png", dpi=300)
    plt.close()

    print(f"🎨 Presentation graphics successfully rendered under: {out_root.resolve()}/")

if __name__ == "__main__":
    # Your standardized clean tracking paths
    HUMAN_GROUND_TRUTH = "datasets/vedai_1024/vedai_color_obb"  # Core filtered human truth set
    PIPELINE_REFINED   = "datasets/vedai_1024/vedai_color_OBB_Refined"    # Your refined pipeline output folder
    EVAL_OUTPUT_DIR    = "visuals/vedai_color_evaluation_results"        # Target diagnostic export directory

    run_evaluation(
        gt_dir=HUMAN_GROUND_TRUTH,
        pred_dir=PIPELINE_REFINED,
        output_dir=EVAL_OUTPUT_DIR,
        iou_threshold=0.50  # Minimum IoU match requirement
    )