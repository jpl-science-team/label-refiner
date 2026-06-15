#!/usr/bin/env python3

import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from segment_anything import sam_model_registry, SamPredictor

# ---------------------------------------------------------
# 1. PerSAM Visual Weights Extractor
# ---------------------------------------------------------
def get_persam_weights(predictor, ref_image, ref_mask):
    """
    Extracts DINOv2 visual features from the reference template 
    so SAM recognizes the custom styling and texture of the cars.
    """
    with torch.no_grad():
        predictor.set_image(ref_image)
        ref_features = predictor.get_image_embedding()
        
        mask_tensor = torch.from_numpy(ref_mask).float().unsqueeze(0).unsqueeze(0).to(ref_features.device)
        mask_resized = F.interpolate(mask_tensor, size=ref_features.shape[-2:], mode="bilinear")
        mask_resized = (mask_resized > 0.5).float()
        
        target_feat = ref_features * mask_resized
        target_weight = target_feat.sum(dim=(-1, -2)) / (mask_resized.sum(dim=(-1, -2)) + 1e-6)
        
    return target_weight

# ---------------------------------------------------------
# 2. Mask → Normalized YOLO OBB Converter (With Geometric Filtering)
# ---------------------------------------------------------
def mask_to_yolo_obb(mask, img_w, img_h):
    """
    Converts binary masks into a 4-point structural Oriented Bounding Box (OBB).
    Includes erosion to separate tightly parked vehicles and defensive filtering 
    to reject giant hallucinations and noise. Adjusted for 1m/pixel spatial resolution.
    """
    # Erode the mask to snap 'bridges' between parked cars, then dilate to restore bounds
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    mask_clean = cv2.erode(mask.astype("uint8"), kernel, iterations=1)
    mask_clean = cv2.dilate(mask_clean, kernel, iterations=1)

    contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    cnt = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(cnt)
    
    # Extract structural dimensions to run verification checks
    (cx, cy), (w, h), angle = rect
    max_dimension = max(w, h)
    min_dimension = min(w, h)
    
    # ---------------------------------------------------------
    # DEFENSIVE FILTERING: Scaled for 1m GSD (1 pixel ≈ 1 meter)
    # ---------------------------------------------------------
    if max_dimension > 12:  # Reject boxes > 12 meters
        return None
        
    if max_dimension < 2:   # Reject boxes < 2 meters
        return None
        
    aspect_ratio = max_dimension / (min_dimension + 1e-6)
    if aspect_ratio > 4.5 or aspect_ratio < 1.1:
        return None
    # ---------------------------------------------------------

    box_points = cv2.boxPoints(rect)
    
    pts = np.array(box_points, dtype="float32")
    x_sorted = pts[np.argsort(pts[:, 0]), :]
    left_pts = x_sorted[:2, :]
    right_pts = x_sorted[2:, :]
    
    tl = left_pts