#!/usr/bin/env python3

import csv
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from segment_anything import sam_model_registry, SamPredictor

# Pixel-scale settings, tuned for COWC 512px tiles with cars ~30px long. Every pixel threshold
# (here and in the functions below) is scaled by CAR_LENGTH / BASE_CAR_LENGTH via px(), so set
# CAR_LENGTH (or --car-length) to the typical car length of the dataset, e.g. 20 for VEDAI 512.
# Snapping and the negative point are off by default: on the COWC mini dataset (accurate point
# labels, ~8px similarity-map resolution) each slightly reduced the number of accepted SAM masks.
USE_NEGATIVE_POINT = False  # add a PerSAM-style negative prompt from the similarity map
NEG_EXCLUDE_RADIUS = 20     # keep the negative point at least this far from any labeled car
SNAP_TO_SIMILARITY_PEAK = False # move each positive prompt to the similarity peak near its label point
SNAP_RADIUS = 8             # max distance (px) a prompt may move when snapping
LOW_SIM_MARGIN = 0.10       # flag boxes whose similarity is less than this above the image's median (background) similarity
REF_CANVAS_SIZE = 512       # a rescaled reference crop is padded to this size, like a dataset tile
DROP_LOW_SIMILARITY = False # if True, flagged boxes are left out of the label files entirely
FEATURES = "sam"            # "sam" (SAM encoder) or "dino" (DINOv2) features for the similarity map
DINO_REPO = "models/dinov2-main"                    # local copy of github.com/facebookresearch/dinov2
DINO_WEIGHTS = "models/dinov2_vits14_pretrain.pth"  # from dl.fbaipublicfiles.com/dinov2/dinov2_vits14/
DINO_UPSCALE = 2            # DINOv2 uses 14px patches; upscaling 512 -> 1024 gives ~7px resolution for ~30px cars
USE_BOX_PROMPT = False      # VS_PerSAM_DINOv2-style prompt: a box + ring of negative points around each car
BOX_HALF_SIZE = 18          # box prompt half-width (px); negative points sit just outside it
CAR_LENGTH = 30             # typical car length (px) in the dataset being refined
BASE_CAR_LENGTH = 30        # car length the pixel thresholds were tuned at (COWC)

def px(value):
    """Scales a pixel threshold tuned for COWC (~30px cars) to the dataset's CAR_LENGTH."""
    return value * CAR_LENGTH / BASE_CAR_LENGTH

# ---------------------------------------------------------
# 1. PerSAM Visual Weights Extractor
# ---------------------------------------------------------
def _valid_features(predictor):
    """
    SAM pads the resized image to a 1024x1024 square, so the 64x64 embedding
    contains padding. Return only the (1, C, fh, fw) region covering real pixels.
    """
    feats = predictor.get_image_embedding()
    in_h, in_w = predictor.input_size
    fh = min(feats.shape[-2], int(np.ceil(in_h / 16)))
    fw = min(feats.shape[-1], int(np.ceil(in_w / 16)))
    return feats[..., :fh, :fw]

class DINOv2FeatureExtractor:
    """
    DINOv2 ViT-S/14 patch features, loaded from a local repo + weights file (no internet needed).
    Mirrors DINOv2FeatureExtractor in VS_PerSAM_DINOv2.py, plus upscaling for small objects.
    """
    def __init__(self, repo_dir=DINO_REPO, weights_path=DINO_WEIGHTS, device="cpu"):
        print(f"Loading DINOv2 ViT-S/14 from {repo_dir} with weights {weights_path}")
        self.device = device
        self.model = torch.hub.load(repo_dir, "dinov2_vits14", source="local", pretrained=False)
        self.model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
        self.model.to(device).eval()
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

    @torch.no_grad()
    def features(self, cv2_img):
        """Returns (1, C, ph, pw) patch features covering the whole image."""
        img_rgb = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).float().to(self.device) / 255.0
        tensor = (tensor - self.mean) / self.std
        h, w = tensor.shape[-2:]
        new_h = ((h * DINO_UPSCALE + 13) // 14) * 14
        new_w = ((w * DINO_UPSCALE + 13) // 14) * 14
        tensor = F.interpolate(tensor, size=(new_h, new_w), mode="bilinear", align_corners=False)
        tokens = self.model.forward_features(tensor)["x_norm_patchtokens"]
        return tokens.reshape(1, new_h // 14, new_w // 14, -1).permute(0, 3, 1, 2)

def image_features(img, predictor, dino=None):
    """
    Feature map used for the similarity map: DINOv2 patch features if `dino` is given,
    otherwise SAM encoder features. Expects predictor.set_image(img) to have been called.
    """
    if dino is not None:
        return dino.features(img)
    return _valid_features(predictor)

def get_persam_weights(ref_features, ref_mask):
    """
    Pools reference features (SAM or DINOv2) over the reference mask and L2-normalizes
    them, giving a (1, C, 1, 1) target signature for cosine-similarity matching.
    """
    with torch.no_grad():
        mask_tensor = torch.from_numpy(ref_mask).float().unsqueeze(0).unsqueeze(0).to(ref_features.device) / 255.0
        mask_resized = F.interpolate(mask_tensor, size=ref_features.shape[-2:], mode="bilinear")
        hard = (mask_resized > 0.5).float()
        if hard.sum() < 1:  # tiny reference: fall back to the soft mask
            hard = mask_resized

        target_feat = (ref_features * hard).sum(dim=(-1, -2), keepdim=True) / (hard.sum() + 1e-6)
        target_feat = F.normalize(target_feat, dim=1)

    return target_feat

def compute_similarity_map(features, target_weight, img_h, img_w):
    """
    Cosine similarity between the target signature and every feature of the image
    (SAM or DINOv2 features from image_features). Returns an (img_h, img_w) numpy array.
    """
    with torch.no_grad():
        feats = F.normalize(features, dim=1)
        sim = (feats * target_weight).sum(dim=1, keepdim=True)
        sim = F.interpolate(sim, size=(img_h, img_w), mode="bilinear")
    return sim.squeeze().cpu().numpy()

def pick_negative_point(similarity_map, positive_points, exclude_radius):
    """
    PerSAM-style negative prompt: the least car-like location in the image,
    ignoring a neighbourhood around every labeled car. Returns [x, y] or None.
    """
    sim = similarity_map.copy()
    h, w = sim.shape
    for px, py in positive_points:
        x0, x1 = max(0, px - exclude_radius), min(w, px + exclude_radius + 1)
        y0, y1 = max(0, py - exclude_radius), min(h, py + exclude_radius + 1)
        sim[y0:y1, x0:x1] = np.inf
    if np.isinf(sim).all():
        return None
    y, x = np.unravel_index(np.argmin(sim), sim.shape)
    return np.array([x, y])

def snap_to_similarity_peak(similarity_map, px, py, radius):
    """
    Moves a label point to the most car-like location within `radius` px,
    correcting point labels that sit slightly off the vehicle. Returns (x, y).
    """
    h, w = similarity_map.shape
    x0, x1 = max(0, px - radius), min(w, px + radius + 1)
    y0, y1 = max(0, py - radius), min(h, py + radius + 1)
    if x0 >= x1 or y0 >= y1:
        return px, py
    window = similarity_map[y0:y1, x0:x1]
    dy, dx = np.unravel_index(np.argmax(window), window.shape)
    return x0 + dx, y0 + dy

# ---------------------------------------------------------
# 2. Mask → Normalized YOLO OBB Converter (With Geometric Filtering)
# ---------------------------------------------------------
def mask_to_yolo_obb(mask, img_w, img_h):
    """
    Converts binary masks into a 4-point structural Oriented Bounding Box (OBB).
    Includes defensive filtering to reject giant hallucinations and noise.
    """
    contours, _ = cv2.findContours(mask.astype("uint8"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    cnt = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(cnt)
    
    # Extract structural dimensions to run verification checks
    (cx, cy), (w, h), angle = rect
    max_dimension = max(w, h)
    min_dimension = min(w, h)
    
    # ---------------------------------------------------------
    # DEFENSIVE FILTERING: Stop massive hallucinations & noise
    # ---------------------------------------------------------
    if max_dimension > px(90):
        return None

    if max_dimension < px(6):
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
    
    tl = left_pts[np.argsort(left_pts[:, 1])[0]]
    bl = left_pts[np.argsort(left_pts[:, 1])[1]]
    tr = right_pts[np.argsort(right_pts[:, 1])[0]]
    br = right_pts[np.argsort(right_pts[:, 1])[1]]
    
    return [
        tl[0] / img_w, tl[1] / img_h,
        tr[0] / img_w, tr[1] / img_h,
        br[0] / img_w, br[1] / img_h,
        bl[0] / img_w, bl[1] / img_h
    ]

# ---------------------------------------------------------
# 3. Post-Processing Geometric Correction Head
# ---------------------------------------------------------
def geometric_correction_head(refined_labels, img_w, img_h):
    """
    STRICT POST-PROCESSING HEAD: Resolves remaining edge case failures 
    by targeting axis-aligned bleeding boxes in building shadows.
    Returns (labels, was_corrected) where was_corrected[i] is True if box i was replaced.
    """
    if len(refined_labels) == 0:
        return refined_labels, []

    corrected_labels = []
    was_corrected = []
    valid_cars = []

    for label in refined_labels:
        cls_id = label[0]
        pts = np.array(label[1:]).reshape(4, 2)
        
        rect = cv2.minAreaRect((pts * np.array([img_w, img_h])).astype(np.float32))
        (cx, cy), (w, h), angle = rect
        max_dim = max(w, h)
        min_dim = min(w, h)
        aspect = max_dim / (min_dim + 1e-6)

        if px(15) < max_dim < px(42) and 1.3 < aspect < 3.2 and angle != 0.0:
            valid_cars.append({
                'center': (cx, cy),
                'size': (w, h),
                'angle': angle,
                'raw_label': label
            })

    for label in refined_labels:
        cls_id = label[0]
        pts = np.array(label[1:]).reshape(4, 2)
        rect = cv2.minAreaRect((pts * np.array([img_w, img_h])).astype(np.float32))
        (cx, cy), (w, h), angle = rect
        max_dim = max(w, h)
        
        is_bad_box = max_dim > px(45) or max_dim < px(8) or angle == 0.0 or angle == 90.0

        if is_bad_box:
            if len(valid_cars) > 0:
                distances = [np.linalg.norm(np.array((cx, cy)) - np.array(vc['center'])) for vc in valid_cars]
                closest_car = valid_cars[np.argmin(distances)]
                matched_w, matched_h = closest_car['size']
                matched_angle = closest_car['angle']
            else:
                matched_w, matched_h = px(30.0), px(15.0)
                matched_angle = 0.0
            
            corrected_rect = ((cx, cy), (matched_w, matched_h), matched_angle)
            box_points = cv2.boxPoints(corrected_rect)
            
            x_sorted = box_points[np.argsort(box_points[:, 0]), :]
            left_pts = x_sorted[:2, :]
            right_pts = x_sorted[2:, :]
            tl = left_pts[np.argsort(left_pts[:, 1])[0]]
            bl = left_pts[np.argsort(left_pts[:, 1])[1]]
            tr = right_pts[np.argsort(right_pts[:, 1])[0]]
            br = right_pts[np.argsort(right_pts[:, 1])[1]]
            
            normalized_obb = [
                tl[0] / img_w, tl[1] / img_h,
                tr[0] / img_w, tr[1] / img_h,
                br[0] / img_w, br[1] / img_h,
                bl[0] / img_w, bl[1] / img_h
            ]
            corrected_labels.append([int(cls_id)] + normalized_obb)
            was_corrected.append(True)
        else:
            corrected_labels.append(label)
            was_corrected.append(False)

    return corrected_labels, was_corrected

# ---------------------------------------------------------
# 4. Load SAM Model Configuration
# ---------------------------------------------------------
def load_sam(model_path="models/sam_vit_b.pth"):
    print(f"Loading SAM checkpoint from: {model_path}")
    sam = sam_model_registry["vit_b"](checkpoint=model_path)

    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
        
    sam.to(device)
    predictor = SamPredictor(sam)
    print(f"SAM loaded successfully on target platform: {device}")
    return predictor, device

# ---------------------------------------------------------
# 5. Per-Folder Refinement (one images/ + labels/ pair)
# ---------------------------------------------------------
def refine_folder(in_dir, out_dir, predictor, target_weight, dino=None, desc="Processing"):
    """
    Refines every image in in_dir/images using its point labels in in_dir/labels,
    writing copied images and OBB labels to out_dir/images and out_dir/labels,
    plus out_dir/refinement_report.csv with one row per box (how it was made,
    its similarity to the reference, and whether it was flagged).
    """
    in_img_dir, in_lbl_dir = in_dir / "images", in_dir / "labels"
    out_img_dir, out_lbl_dir = out_dir / "images", out_dir / "labels"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    image_list = sorted([f for f in os.listdir(in_img_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))])

    if len(image_list) == 0:
        print(f"[WARNING] No images (.png, .jpg) found in {in_img_dir}")
        return 0

    processed_count = 0
    report_rows = []

    for img_name in tqdm(image_list, desc=desc):
        img_path = in_img_dir / img_name
        lbl_path = in_lbl_dir / (Path(img_name).stem + ".txt")

        if not lbl_path.exists():
            print(f"[WARNING] Skipping {img_name}: Missing label file at {lbl_path}")
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[WARNING] Skipping {img_name}: Unable to read image file.")
            continue

        img_h, img_w = img.shape[:2]
        cv2.imwrite(str(out_img_dir / img_name), img)

        with open(lbl_path) as f:
            lines = f.read().strip().splitlines()

        refined_labels = []
        predictor.set_image(img)

        similarity_map = compute_similarity_map(image_features(img, predictor, dino), target_weight, img_h, img_w)

        objs = []
        for line in lines:
            parts = list(map(float, line.split()))
            if len(parts) < 5:
                continue
            objs.append(parts[:5])

        positive_points = [(int(round(cx * img_w)), int(round(cy * img_h))) for _, cx, cy, _, _ in objs]
        neg_point = None
        if USE_NEGATIVE_POINT and positive_points:
            neg_point = pick_negative_point(similarity_map, positive_points, int(round(px(NEG_EXCLUDE_RADIUS))))

        sources, similarities, prompts = [], [], []
        # Similarity is judged relative to this image's typical (mostly background) value, which
        # keeps the flag meaningful regardless of which reference or image conditions are used
        background_sim = float(np.median(similarity_map))

        for (cls_id, cx, cy, w, h), (lx, ly) in zip(objs, positive_points):
            if SNAP_TO_SIMILARITY_PEAK:
                x, y = snap_to_similarity_peak(similarity_map, lx, ly, int(round(px(SNAP_RADIUS))))
            else:
                x, y = lx, ly
            prompts.append((x, y))
            similarities.append(float(similarity_map[min(y, img_h - 1), min(x, img_w - 1)]))

            input_box = None
            if USE_BOX_PROMPT:
                # As in VS_PerSAM_DINOv2.py: one positive at the car, a box around it, and a ring of
                # negatives just outside the box so the mask can't bleed into neighbours or shadows
                half = px(BOX_HALF_SIZE)
                r = half + 1
                input_points = np.array([
                    [x, y],
                    [x - r, y], [x + r, y], [x, y - r], [x, y + r],
                    [x - r, y - r], [x + r, y - r], [x - r, y + r], [x + r, y + r]
                ])
                input_labels = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0])
                input_box = np.array([[x - half, y - half, x + half, y + half]])
            else:
                nudge = px(4)
                input_points = np.array([
                    [x, y],
                    [x - nudge, y],
                    [x + nudge, y],
                    [x, y - nudge],
                    [x, y + nudge]
                ])
                input_labels = np.array([1, 1, 1, 1, 1])

            if neg_point is not None:
                input_points = np.vstack([input_points, neg_point])
                input_labels = np.append(input_labels, 0)

            masks, scores, _ = predictor.predict(
                point_coords=input_points,
                point_labels=input_labels,
                box=input_box,
                multimask_output=False
            )

            mask = masks[0]
            yolo_obb = mask_to_yolo_obb(mask, img_w, img_h)
            sources.append("sam" if yolo_obb is not None else "fallback")

            if yolo_obb is None:
                # Placeholder box stays centered on the original label point
                x1_n = cx - (w / 2.0)
                y1_n = cy - (h / 2.0)
                x2_n = cx + (w / 2.0)
                y2_n = cy + (h / 2.0)

                yolo_obb = [x1_n, y1_n, x2_n, y1_n, x2_n, y2_n, x1_n, y2_n]

            refined_labels.append([int(cls_id)] + yolo_obb)

        final_clean_labels, was_corrected = geometric_correction_head(refined_labels, img_w, img_h)

        out_lbl_path = out_lbl_dir / (Path(img_name).stem + ".txt")
        with open(out_lbl_path, "w") as f:
            for i, r in enumerate(final_clean_labels):
                contrast = similarities[i] - background_sim
                low_sim = contrast < LOW_SIM_MARGIN
                kept = not (DROP_LOW_SIMILARITY and low_sim)
                if kept:
                    coords_str = " ".join(f"{c:.6f}" for c in r[1:])
                    f.write(f"{int(r[0])} {coords_str}\n")

                report_rows.append({
                    "image": img_name,
                    "box": i,
                    "source": "corrected" if was_corrected[i] else sources[i],
                    "similarity": f"{similarities[i]:.4f}",
                    "background_similarity": f"{background_sim:.4f}",
                    "contrast": f"{contrast:.4f}",
                    "low_similarity": int(low_sim),
                    "kept": int(kept),
                    "label_x": positive_points[i][0],
                    "label_y": positive_points[i][1],
                    "prompt_x": prompts[i][0],
                    "prompt_y": prompts[i][1],
                })

        processed_count += 1

    # source: sam = SAM mask accepted; fallback = mask rejected, original placeholder box kept;
    # corrected = replaced by the geometric correction head (size/angle borrowed from a nearby car)
    with open(out_dir / "refinement_report.csv", "w", newline="") as f:
        fields = ["image", "box", "source", "similarity", "background_similarity", "contrast", "low_similarity", "kept",
                  "label_x", "label_y", "prompt_x", "prompt_y"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(report_rows)

    n_low = sum(r["low_similarity"] for r in report_rows)
    by_source = {s: sum(r["source"] == s for r in report_rows) for s in ("sam", "fallback", "corrected")}
    print(f"    Boxes: {len(report_rows)} ({by_source['sam']} sam, {by_source['fallback']} fallback, "
          f"{by_source['corrected']} corrected), {n_low} flagged (contrast < {LOW_SIM_MARGIN})"
          + (" (dropped)" if DROP_LOW_SIMILARITY else ""))

    return processed_count

# ---------------------------------------------------------
# 6. Reference Construction
# ---------------------------------------------------------
SPLITS = ["train", "val", "test"]

def read_label_points(lbl_path, img_w, img_h):
    points = []
    for line in Path(lbl_path).read_text().splitlines():
        parts = line.split()
        if len(parts) >= 5:
            points.append((int(round(float(parts[1]) * img_w)), int(round(float(parts[2]) * img_h))))
    return points

def find_reference_tile(input_dir):
    """Picks the image with the most point labels (preferring the train split) as the reference tile."""
    input_dir = Path(input_dir)
    if (input_dir / "images").is_dir():
        folders = [input_dir]
    else:
        folders = [input_dir / s for s in SPLITS if (input_dir / s / "images").is_dir()]
    for folder in folders:
        best, best_count = None, 0
        for img_path in sorted((folder / "images").iterdir()):
            lbl_path = folder / "labels" / (img_path.stem + ".txt")
            if img_path.suffix.lower() in (".png", ".jpg", ".jpeg") and lbl_path.exists():
                count = sum(1 for l in lbl_path.read_text().splitlines() if len(l.split()) >= 5)
                if count > best_count:
                    best, best_count = img_path, count
        if best is not None:
            return best
    raise FileNotFoundError(f"No labeled images found in {input_dir} to use as a reference tile")

def build_reference_from_tile(predictor, img, points):
    """
    PerSAM-style reference: segments every labeled car in one tile with SAM and returns
    the union of the masks that pass the OBB size/aspect checks. Using a tile from the
    dataset keeps the reference at the same scale and imaging conditions as the targets.
    """
    img_h, img_w = img.shape[:2]
    predictor.set_image(img)
    ref_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    used = 0
    for x, y in points:
        masks, _, _ = predictor.predict(point_coords=np.array([[x, y]]), point_labels=np.array([1]),
                                        multimask_output=False)
        contours, _ = cv2.findContours(masks[0].astype("uint8"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        _, (w, h), _ = cv2.minAreaRect(max(contours, key=cv2.contourArea))
        aspect = max(w, h) / (min(w, h) + 1e-6)
        # Same single-car criteria as the correction head's "valid car" check, so merged
        # blobs of several cars + shadow don't leak asphalt into the signature
        if px(15) < max(w, h) < px(42) and 1.3 < aspect < 3.2:
            ref_mask[masks[0]] = 255
            used += 1
    if used == 0:
        raise RuntimeError("No usable car masks in the reference tile; pick another with --ref-tile")
    return ref_mask, used

def estimate_upscale_factor(img):
    """
    Detects nearest-neighbour upscaling (e.g. a 30px car crop blown up 10x) from the width
    of runs of identical adjacent columns. Returns 1 for an image at native resolution.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(int)
    changes = np.flatnonzero(np.any(np.diff(gray, axis=1) != 0, axis=0))
    if len(changes) < 2:
        return 1
    return max(1, int(round(np.median(np.diff(changes)))))

def load_reference_crop(predictor, ref_img_path, ref_mask_path=None, scale=None):
    """
    Reference from a hand-cropped car image. The crop is shrunk back to native resolution
    (auto-detected unless `scale` is given) and padded to a tile-sized canvas so SAM sees the
    car at the same scale as in the dataset. The car is segmented with SAM at the crop centre
    unless a mask file is given. Returns (canvas, mask).
    """
    ref_img = cv2.imread(str(ref_img_path))
    if ref_img is None:
        raise FileNotFoundError(f"Reference crop missing or unreadable: {ref_img_path}")
    if scale is None:
        scale = 1.0 / estimate_upscale_factor(ref_img)

    small = cv2.resize(ref_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    h, w = small.shape[:2]
    if max(h, w) > REF_CANVAS_SIZE:
        raise ValueError(f"Rescaled reference ({w}x{h}) is larger than the {REF_CANVAS_SIZE}px canvas; pass --ref-scale")
    top, left = (REF_CANVAS_SIZE - h) // 2, (REF_CANVAS_SIZE - w) // 2
    pad = (top, REF_CANVAS_SIZE - h - top, left, REF_CANVAS_SIZE - w - left)
    canvas = cv2.copyMakeBorder(small, *pad, cv2.BORDER_REPLICATE)

    if ref_mask_path:
        mask = cv2.imread(str(ref_mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Reference mask missing or unreadable: {ref_mask_path}")
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        mask = cv2.copyMakeBorder(mask, *pad, cv2.BORDER_CONSTANT, value=0)
    else:
        predictor.set_image(canvas)
        masks, _, _ = predictor.predict(point_coords=np.array([[REF_CANVAS_SIZE // 2, REF_CANVAS_SIZE // 2]]),
                                        point_labels=np.array([1]), multimask_output=False)
        mask = masks[0].astype(np.uint8) * 255

    contours, _ = cv2.findContours((mask > 127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    length = max(cv2.minAreaRect(max(contours, key=cv2.contourArea))[1]) if contours else 0
    print(f"Reference crop: {ref_img_path} scaled x{scale:.3g} to {w}x{h}px, car mask {length:.0f}px long")
    if not px(15) < length < px(42):
        print(f"[WARNING] Reference car mask is outside the expected {px(15):.0f}-{px(42):.0f}px car length; "
              "check reference_mask.png in the output folder, or set --ref-scale / --ref-mask")
    return canvas, mask

# ---------------------------------------------------------
# 7. Dataset Processing (flat or train/val/test layout, with YAML generation)
# ---------------------------------------------------------
def refine_dataset_persam(input_dir, output_dir, predictor, ref_img, ref_mask, dino=None):
    """
    input_dir can be either:
      - flat:   input_dir/images + input_dir/labels
      - splits: input_dir/{train,val,test}/images + labels (missing splits are skipped)
    The output mirrors the input layout.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save the reference with its mask outlined in green, for checking what PerSAM matched against
    ref_viz = ref_img.copy()
    contours, _ = cv2.findContours((ref_mask > 127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(ref_viz, contours, -1, (0, 255, 0), 1)
    cv2.imwrite(str(output_dir / "reference_mask.png"), ref_viz)

    print(f"Extracting PerSAM target signature from {'DINOv2' if dino is not None else 'SAM encoder'} features...")
    predictor.set_image(ref_img)
    target_weight = get_persam_weights(image_features(ref_img, predictor, dino), ref_mask)

    print(f"\n=== Running PerSAM Refinement: {input_dir} -> {output_dir} ===")

    if (input_dir / "images").is_dir():
        count = refine_folder(input_dir, output_dir, predictor, target_weight, dino)
        print(f"    Successfully processed {count} images.")
        print(f"✔ Dataset processing complete: {input_dir}")
        return

    found_splits = [s for s in SPLITS if (input_dir / s / "images").is_dir()]
    if not found_splits:
        raise FileNotFoundError(f"No images/ folder found in {input_dir} or its {'/'.join(SPLITS)} subfolders")

    for split in found_splits:
        print(f"--> Processing split: {split} from {input_dir / split / 'images'}")
        count = refine_folder(input_dir / split, output_dir / split, predictor, target_weight, dino, desc=f"Processing {split}")
        print(f"    Successfully processed {count} images for split '{split}'.")

    print(f"✔ Dataset processing complete: {input_dir}")

    # ---------------------------------------------------------
    # Generate YAML configuration file for YOLO training
    # ---------------------------------------------------------
    split_lines = "".join(f"{s}: {s}/images\n" for s in found_splits)
    yaml_content = f"""path: {output_dir.absolute()}
{split_lines}
# Classes
names:
  0: car
"""
    yaml_path = output_dir / "data.yaml"
    with open(yaml_path, "w") as yaml_file:
        yaml_file.write(yaml_content)
    print(f"✔ Generated YOLO config file at: {yaml_path}")

# ---------------------------------------------------------
# Entry Point
# ---------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Refine YOLO point labels into OBB labels with PerSAM.")
    parser.add_argument("--input", default="datasets/COWC_Points_512",
                        help="Dataset dir: either images/+labels/ or train|val|test/{images,labels}")
    parser.add_argument("--output", default="datasets_refined/COWC_512", help="Output dir (mirrors input layout)")
    parser.add_argument("--ref", default=None,
                        help="Reference car crop (default: <input>/ref_car.png if it exists, otherwise --ref-tile is used)")
    parser.add_argument("--ref-scale", type=float, default=None,
                        help="Resize factor for --ref (default: undo detected upscaling, e.g. 0.1 for a 10x blown-up crop)")
    parser.add_argument("--ref-mask", default=None, help="Car mask for --ref (default: segment the crop with SAM)")
    parser.add_argument("--ref-tile", default=None,
                        help="Labeled image to build the reference from when no crop is used "
                             "(default: the input image with the most labels)")
    parser.add_argument("--model", default="models/sam_vit_b.pth", help="SAM ViT-B checkpoint")
    parser.add_argument("--negative", action="store_true", help="Add a similarity-based negative prompt")
    parser.add_argument("--snap", action="store_true", help="Move each prompt to the similarity peak near its label point")
    parser.add_argument("--sim-margin", type=float, default=LOW_SIM_MARGIN,
                        help=f"Flag boxes whose similarity is less than this above the image's median similarity "
                             f"(default {LOW_SIM_MARGIN})")
    parser.add_argument("--drop-low-sim", action="store_true", help="Leave flagged boxes out of the label files")
    parser.add_argument("--features", choices=["sam", "dino"], default=FEATURES,
                        help=f"Features for the similarity map (default {FEATURES})")
    parser.add_argument("--box-prompt", action="store_true",
                        help="Prompt SAM with a box + ring of negative points around each car (as in VS_PerSAM_DINOv2.py)")
    parser.add_argument("--car-length", type=float, default=CAR_LENGTH,
                        help=f"Typical car length in px; all pixel thresholds scale with it (default {CAR_LENGTH}, "
                             f"COWC; use 20 for VEDAI 512)")
    parser.add_argument("--box-half-size", type=float, default=None,
                        help=f"Box prompt half-width in px (default {BOX_HALF_SIZE} scaled by --car-length)")
    args = parser.parse_args()

    USE_NEGATIVE_POINT = USE_NEGATIVE_POINT or args.negative
    SNAP_TO_SIMILARITY_PEAK = SNAP_TO_SIMILARITY_PEAK or args.snap
    LOW_SIM_MARGIN = args.sim_margin
    DROP_LOW_SIMILARITY = args.drop_low_sim
    USE_BOX_PROMPT = USE_BOX_PROMPT or args.box_prompt
    CAR_LENGTH = args.car_length
    if args.box_half_size is not None:
        # px() scales BOX_HALF_SIZE again when it is used, so store the unscaled equivalent
        BOX_HALF_SIZE = args.box_half_size * BASE_CAR_LENGTH / CAR_LENGTH
    print(f"Car length {CAR_LENGTH:g}px: pixel thresholds scaled x{CAR_LENGTH / BASE_CAR_LENGTH:.3g}"
          + (f", box prompt {2 * px(BOX_HALF_SIZE):.0f}px" if USE_BOX_PROMPT else ""))

    sam_predictor, dev_env = load_sam(args.model)
    dino_extractor = DINOv2FeatureExtractor(device=dev_env) if args.features == "dino" else None

    ref_crop = Path(args.ref) if args.ref else Path(args.input) / "ref_car.png"
    if args.ref or (ref_crop.exists() and not args.ref_tile):
        ref_img, ref_mask = load_reference_crop(sam_predictor, ref_crop, args.ref_mask, args.ref_scale)
    else:
        ref_tile = Path(args.ref_tile) if args.ref_tile else find_reference_tile(args.input)
        ref_lbl = ref_tile.parent.parent / "labels" / (ref_tile.stem + ".txt")
        ref_img = cv2.imread(str(ref_tile))
        if ref_img is None or not ref_lbl.exists():
            raise FileNotFoundError(f"Reference tile or its labels missing:\n {ref_tile}\n {ref_lbl}")
        ref_points = read_label_points(ref_lbl, ref_img.shape[1], ref_img.shape[0])
        ref_mask, n_used = build_reference_from_tile(sam_predictor, ref_img, ref_points)
        print(f"Reference tile: {ref_tile} ({n_used}/{len(ref_points)} labeled cars used)")

    refine_dataset_persam(
        input_dir=args.input,
        output_dir=args.output,
        predictor=sam_predictor,
        ref_img=ref_img,
        ref_mask=ref_mask,
        dino=dino_extractor
    )
