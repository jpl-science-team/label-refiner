#!/usr/bin/env python3

import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from shapely.geometry import Polygon

# =========================================================
# VEDAI Target Classes
# Target (Evaluated): 1 = Car, 2 = Pickup
# Ignored (Not Penalized): 4 = Truck, 5 = Semi, 9 = Van, etc.
# =========================================================
TARGET_CLASSES = {1, 2} 

def obb_to_polygon(obb_coords, is_normalized=True, img_w=1024.0, img_h=1024.0):
    """Converts OBB coordinates to a Shapely Polygon."""
    pts = np.array(obb_coords).reshape(4, 2)
    
    # YOLO format is normalized, raw VEDAI is absolute
    if is_normalized:
        pts[:, 0] *= img_w
        pts[:, 1] *= img_h
        
    poly = Polygon(pts)
    if not poly.is_valid:
        poly = poly.buffer(0)
    return poly

def compute_obb_iou(poly1, poly2):
    """Computes exact Intersection over Union between oriented polygons."""
    try:
        inter_area = poly1.intersection(poly2).area
        union_area = poly1.area + poly2.area - inter_area
        return inter_area / (union_area + 1e-6)
    except Exception:
        return 0.0

def load_raw_vedai_gt(file_path):
    """
    Parses original VEDAI 14-column labels.
    Assigns eval_class 0 to targets, and -1 to ignored classes to avoid penalizing them.
    """
    gt_items = []
    if not file_path.exists():
        return gt_items
    
    with open(file_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            try:
                if len(parts) >= 14:
                    orig_class = int(parts[3])
                    
                    # Strict target matching[cite: 1]
                    eval_class = 0 if orig_class in TARGET_CLASSES else -1
                    
                    # Extract absolute pixel coordinates[cite: 1]
                    x_coords = [float(x) for x in parts[6:10]]
                    y_coords = [float(y) for y in parts[10:14]]
                    coords = [
                        x_coords[0], y_coords[0],
                        x_coords[1], y_coords[1],
                        x_coords[2], y_coords[2],
                        x_coords[3], y_coords[3]
                    ]
                    
                    poly = obb_to_polygon(coords, is_normalized=False)
                    gt_items.append({'class': eval_class, 'poly': poly})
            except (ValueError, IndexError):
                continue
    return gt_items

def load_pipeline_preds(file_path):
    """Parses pipeline refined YOLO OBB (9-column) text files."""
    preds = []
    if not file_path.exists():
        return preds
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 9:
                continue
            coords = [float(x) for x in parts[1:9]]
            poly = obb_to_polygon(coords, is_normalized=True)
            preds.append(poly)
    return preds

def run_evaluation(raw_gt_dir, pred_dir, output_dir, iou_threshold=0.50):
    raw_gt_root = Path(raw_gt_dir)
    pred_root = Path(pred_dir)
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    pred_lbl_dir = pred_root / "labels" if (pred_root / "labels").exists() else pred_root
    pred_files = list(pred_lbl_dir.glob("*.txt"))
    
    print(f"🧐 Found {len(pred_files)} pipeline prediction files. Cross-referencing raw VEDAI data...")

    confusion_data = {"Success": 0, "Pipeline_Failure": 0, "Ignored_Targets_Hit": 0}
    all_ious = []

    for pred_path in pred_files:
        # Map pipeline name (00000001_co.txt) back to raw VEDAI annotation (00000001.txt)
        raw_gt_stem = pred_path.stem.split('_')[0] 
        gt_path = raw_gt_root / f"{raw_gt_stem}.txt"
        
        gt_boxes = load_raw_vedai_gt(gt_path)
        pred_polys = load_pipeline_preds(pred_path)
        
        # Isolate targets (Cars/Pickups) from ignored classes (Trucks/Semis)[cite: 1]
        valid_gts = [gt['poly'] for gt in gt_boxes if gt['class'] == 0]
        ignore_gts = [gt['poly'] for gt in gt_boxes if gt['class'] == -1]
        
        # 1. Filter predictions hitting ignored (DontCare) targets[cite: 1]
        valid_preds = []
        for p_poly in pred_polys:
            hit_ignore = False
            for ig_poly in ignore_gts:
                if compute_obb_iou(p_poly, ig_poly) > iou_threshold:
                    hit_ignore = True
                    confusion_data["Ignored_Targets_Hit"] += 1
                    break
            
            # Only keep predictions that don't overlap with a truck/semi[cite: 1]
            if not hit_ignore:
                valid_preds.append(p_poly)
                
        # 2. Match remaining valid predictions against strict Target Cars/Pickups
        matched_preds = set()
        for v_gt in valid_gts:
            best_iou = 0.0
            best_idx = -1
            
            # Greedy matching for target polygons[cite: 1]
            for idx, v_pred in enumerate(valid_preds):
                if idx in matched_preds:
                    continue
                iou = compute_obb_iou(v_gt, v_pred)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx
                    
            # Check if prediction meets overlap threshold for success
            if best_iou >= iou_threshold and best_idx != -1:
                confusion_data["Success"] += 1
                matched_preds.add(best_idx)
                all_ious.append(best_iou)
            else:
                # PENALIZED: Pipeline failed SAM on a valid car, or filtered it by mistake
                confusion_data["Pipeline_Failure"] += 1

    # Compute Output Metrics
    successes = confusion_data["Success"]
    failures = confusion_data["Pipeline_Failure"]
    ignored_hit = confusion_data["Ignored_Targets_Hit"]
    total_valid_targets = successes + failures
    
    mean_iou = np.mean(all_ious) if all_ious else 0.0
    success_rate = (successes / total_valid_targets * 100) if total_valid_targets > 0 else 0.0

    # Write Diagnostics Report
    report_path = out_root / "pipeline_evaluation_report.txt"
    with open(report_path, "w") as f:
        f.write("==================================================\n")
        f.write("   SAM + DINO STRICT TARGET EVALUATION REPORT\n")
        f.write("==================================================\n")
        f.write(f"Raw Annotations Dir    : {raw_gt_root.resolve()}\n")
        f.write(f"Pipeline Prediction Dir: {pred_lbl_dir.resolve()}\n")
        f.write(f"IoU Matching Threshold : {iou_threshold}\n")
        f.write("--------------------------------------------------\n")
        f.write(f"Total Target Vehicles (Cars/Pickups) : {total_valid_targets}\n")
        f.write(f"Pipeline Successes (TP)              : {successes}\n")
        f.write(f"Pipeline Failures (Penalized FN)     : {failures}\n")
        f.write(f"Ignored Objects Detected (No Penalty): {ignored_hit}\n")
        f.write("--------------------------------------------------\n")
        f.write(f"Strict Target Alignment Accuracy     : {success_rate:.2f}%\n")
        f.write(f"Mean Oriented IoU (Success)          : {mean_iou:.4f}\n")
        f.write("==================================================\n")

    print(f"\n📊 Performance Metrics Logged to: {report_path}")

    # Render Sleek Presentation Chart
    data = [successes, failures]
    colors = ['#2A75D3', '#EF5350'] # Corporate Steel Blue & Soft Coral
    
    fig, ax = plt.subplots(figsize=(8, 5.5), facecolor='white')
    
    wedges, texts, autotexts = ax.pie(
        data, 
        colors=colors, 
        autopct='%1.1f%%', 
        startangle=90, 
        pctdistance=0.5,
        textprops={'fontsize': 14, 'fontweight': 'bold', 'color': 'white'},
        wedgeprops={'edgecolor': 'white', 'linewidth': 2}
    )
    
    plt.title('SAM + DINO Target Vehicle Pipeline Success\n(Excludes Trucks, Semis, Vans)', 
              fontsize=16, fontweight='bold', pad=15, color='#333333', loc='center')

    legend_labels = [
        f"Success (IoU $\geq$ {iou_threshold})\n{successes} boxes", 
        f"Pipeline Failure (Cars Only)\n{failures} boxes"
    ]
    ax.legend(
        wedges, 
        legend_labels,
        title="Evaluated Outcomes",
        title_fontproperties={'weight':'bold', 'size': 12},
        loc="center left", 
        bbox_to_anchor=(1, 0.5),
        frameon=False,
        fontsize=11,
        labelspacing=1.2
    )
    
    plt.tight_layout()
    plt.savefig(out_root / "alignment_success_pie_chart.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"🎨 Presentation graphics rendered to: {out_root.resolve()}/")

if __name__ == "__main__":
    # Point directly to RAW Annotations to retain VEDAI class IDs
    RAW_VEDAI_ANNOTATIONS = "data/VEDAI/Annotations1024"
    PIPELINE_REFINED      = "datasets/vedai_1024/vedai_color_OBB_Refined"
    EVAL_OUTPUT_DIR       = "visuals/vedai_color_evaluation_results"

    run_evaluation(
        raw_gt_dir=RAW_VEDAI_ANNOTATIONS,
        pred_dir=PIPELINE_REFINED,
        output_dir=EVAL_OUTPUT_DIR,
        iou_threshold=0.50
    )