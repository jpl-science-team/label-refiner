# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Pipeline for refining small-object (vehicle) labels on aerial imagery (COWC, VEDAI) into YOLO-style Oriented Bounding Box (OBB) labels, using SAM plus a one-shot reference crop (`ref_car.png`). `VS_PerSAM_DINOv2.py` (VEDAI) downloads DINOv2 via `torch.hub`, which fails on analysis.jpl.nasa.gov (the proxy blocks github.com). The COWC script uses SAM encoder features by default and DINOv2 with `--features dino`, loaded from local, git-ignored files `models/dinov2-main/` and `models/dinov2_vits14_pretrain.pth` (see README). The repo is under development: Option A (VEDAI study) is in progress; Option B (COWC) works. There is no build system, test suite, or linter — everything is a standalone script.

## Setup

- Python 3.10 in a conda env (conda-forge channel only): `conda create -n label-refiner python=3.10 -y && conda activate label-refiner && pip install -r requirements.txt`
- Git LFS is required (`.gitattributes` tracks png/jpg/txt/pth and `datasets_refined/**`). Run `git lfs install` before cloning/pulling.
- Raw data (`data/COWC`, `data/VEDAI`) is not in the repo; it is downloaded from a Google Drive link in README.md and unpacked per the README's commands.
- `Personalize-SAM/` is a vendored copy of the PerSAM project (own README/requirements); the `scripts/` code imports `segment_anything` (pip package), not `per_segment_anything`.

## Running

**Always run from the repo root** — scripts use relative paths (`data/`, `datasets/`, `datasets_refined/`, `models/sam_vit_b.pth`) and create output dirs themselves. Paths and parameters are mostly hardcoded as constants at the top of each script; edit them in place. Exception: `01cowc_persam_generate_dataset.py` takes CLI options (`--input`, `--output`, `--ref`, `--ref-tile`, `--box-prompt`, `--features sam|dino`, `--snap`, `--negative`, `--sim-margin`, ...; see README "Refining a small dataset") and accepts either a flat `images/`+`labels/` dir or a `train|val|test/{images,labels}` dir, mirroring the layout in its output (`data.yaml` only for the split layout). Example: `python scripts/01cowc_persam_generate_dataset.py --input mini_dataset --output mini_dataset_refined --ref datasets/COWC_Points_512/ref_car.png`. Plot OBB labels over images with `data_exploration/plot_obb_labels.py`, and compare runs side by side with `data_exploration/compare_plots_html.py` (static page, auto-reloads; README covers viewing it over an SSH tunnel). Use the `label-refiner` conda env; on analysis.jpl.nasa.gov CUDA is unusable (driver too old), so SAM runs on CPU at about 3 s per 512px tile.

Option A (VEDAI): `python scripts/VS_VEDAI_Separation_Conversion.py` → `python scripts/VS_PerSAM_DINOv2.py` → `python scripts/VS_evaluate_pipeline.py` (results land in `visuals/vedai_*_evaluation_results/`). Defaults to the color (RGB) dataset.

Option B (COWC): `python scripts/00pure_geographic_splitter.py` → `python scripts/01cowc_persam_generate_dataset.py` → `python scripts/02sensor_degradation.py` → `python data_exploration/val_yolo_dataset.py`.

## Architecture

Stages hand off through directories:

1. `data/` (raw) → **splitter/converter** scripts (`00*`, `VS_VEDAI_Separation_Conversion.py`) tile into 512x512 patches (stride 256) with geographic train/val/test splits, producing point-style (center-point) YOLO labels in `datasets/COWC_Points_512/`.
2. **Refinement** (`01cowc_persam_generate_dataset.py`, `VS_PerSAM_DINOv2.py`): loads SAM ViT-B, computes a target feature vector from a reference, prompts SAM at each labeled point, converts masks to 4-point OBBs via `cv2.minAreaRect`, then applies geometric filtering (reject boxes with max dimension >90 px or <6 px, duplicates by IoU, etc.). Writes to `datasets_refined/COWC_512/` with `data.yaml`.

   In the COWC script the reference only affects the per-box `refinement_report.csv` flags (cosine similarity vs. the image's median), not the label geometry, unless `--snap`/`--negative` are passed; both are off by default because they reduced accepted SAM masks on the mini dataset. `--box-prompt` (box + ring of negatives, from the VS script, but keeping fallback boxes rather than dropping cars) is the biggest improvement found so far: 54/65 accepted masks vs 24. DINOv2 features only separate cars from background with a reference *tile*; with the padded `ref_car.png` crop they don't (they depend on context), whereas SAM features work with either. The reference must match the dataset's scale: an upscaled `ref_car.png` is shrunk back automatically (pixel-block detection) and padded onto a 512 canvas, and its mask comes from SAM. Never use Otsu masks for the reference — on COWC they came out inverted (pavement marked as car). All size thresholds are in pixels, tuned for COWC (~30px cars). `mask_to_yolo_obb` and `load_sam` are duplicated across the COWC and VEDAI scripts — keep them in sync if changing one.
3. `02sensor_degradation.py` applies sensor artifacts/resolution degradation to build a degraded copy of the dataset.
4. `data_exploration/` holds interactive inspection/QA tools (OpenCV-based; `val_yolo_dataset.py` uses A/D to navigate and M to flag bad images; the split being validated is set in a config at the top of the file). Others produce the visuals in `visuals/`.

`VS_*` scripts are the VEDAI study (DINOv2 → SAM relabeling and precision/recall evaluation); un-prefixed numbered scripts are the COWC path.
