# Copyright Statement

Copyright 2026, by the California Institute of Technology. ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged. Any commercial use must be negotiated with the Office of Technology Transfer at the California Institute of Technology.
 
This software may be subject to U.S. export control laws. By accepting this software, the user agrees to comply with all applicable U.S. export laws and regulations. User has the responsibility to obtain export licenses, or other export authority as may be required before exporting such information to foreign countries or providing access to foreign persons.

# About
Label Refiner & Small Object Detection Tools
This repository provides an automated pipeline for preparing, exploring, and refining small-object datasets—specifically targeting COWC variants, VEDAI, and other aerial imagery sources.

The primary objective is to generate clean, consistent YOLO-style Oriented Bounding Box (OBB) labels by leveraging the Segment Anything Model (SAM) and Personalized SAM (PerSAM) for bounding box refinement.

# Directory

🌴 Directory Structure
Use this file tree to check your directory setup.

label-refiner/

├── README.md

├── requirements.txt

├── ref_car.png                     <-- Reference vehicle crop placed here initially

├── models/

│   └── sam_vit_b.pth               <-- Included pre-trained SAM checkpoint

├── data/                           <-- Extracted raw datasets

│   ├── COWC/

│   └── VEDAI/

├── datasets/                       <-- Pipeline staging area

│   └── COWC_Points_512/

│       ├── ref_car.png             <-- Moved here before running PerSAM

│       ├── train/

│       ├── val/

│       └── test/

├── datasets_refined/               <-- Generated output datasets

│   └── COWC_512/

│       ├── data.yaml

│       ├── train/

│       ├── val/

│       └── test/

├── scripts/                        <-- Processing scripts

│   ├── 00cowc_preprocess.py

│   ├── 00pure_geographic_splitter.py

│   ├── 01cowc_persam_generate_dataset.py   <-- older Option B script

│   ├── refine_labels.py                    <-- Option 0

│   ├── evaluate_updated_vedai_labels.py    <-- Option 0: score against VEDAI truth

│   ├── make_cowc_subset.py                 <-- Option 0: rebuild the COWC mini dataset

│   ├── make_vedai_subset.py                <-- Option 0: build VEDAI test sets

│   ├── run_vedai_sweep.sh                  <-- Option 0: parameter sweep

│   ├── summarize_vedai_sweep.py            <-- Option 0: sweep table + comparison page

│   ├── 02sensor_degradation.py

│   ├── VS_VEDAI_Separation_Conversion.py   <-- also prepares VEDAI for Option 0

│   ├── VS_PerSAM_DINOv2.py                 <-- superseded by Option 0

│   └── VS_evaluate_pipeline.py             <-- superseded by Option 0

├── COWC_mini_test/                 <-- COWC mini dataset and refinement runs (image_list.txt; rebuild with make_cowc_subset.py)

├── VEDAI_mini_test/                <-- VEDAI quick-check set (image_list.txt; rebuild with make_vedai_subset.py)

└── data_exploration/               <-- Inspection and validation tools

    ├── plot_obb_labels.py

    ├── compare_plots_html.py

    └── val_yolo_dataset.py

🛠️ Complete Beginner's Setup Guide

If you have never used Conda or Terminal before, follow these step-by-step commands in order.

1. Install Miniforge3 (Conda Environment Manager)
If you do not have Conda installed, run the following commands in your terminal:

On macOS (using Homebrew):

```bash
brew install miniforge
conda init "$(basename "$SHELL")"
#(After running conda init, close and reopen your terminal window).
```

For linux, you'll have to download and install in your environment.

2. Clone Repository & Setup Conda Environment
Copy and paste these commands into your terminal to clone the code and build an isolated environment with Python 3.10:

# Clone the repo and set up the environment

## 0. Make sure you have git lfs

This repo uses lfs, so make sure you have it installed and initialized:

```bash
# for mac
brew install git-lfs
# for linux, you'll have to download: https://github.com/git-lfs/git-lfs?utm_source=gitlfs_site&utm_medium=installation_link&utm_campaign=gitlfs#installing

# next initialize:
git lfs install 
```

## 1. Clone the repository and enter the folder

```bash
git clone https://github.jpl.nasa.gov/science-team-algorithms/label-refiner.git
cd label-refiner
```
## 2. Create the Conda environment

```bash
conda create -n label-refiner python=3.10 -y
```

If you run into anaconda errors (license not supported), you'll want to make sure you use only conda-forge (no defaults or anaconda) in your channels, and your .condarc file should look something like:

ssl_verify: true
channels:
  - conda-forge
env_prompt: ({name})
channel_priority: strict
 
## 3. Activate the environment
```bash
conda activate label-refiner
```
## 4. Install all required dependencies
```bash
pip install -r requirements.txt
```
## 5. Setup Reference Image Artifact
Move the reference image (ref_car.png) into the target dataset directory where PerSAM expects to find it.

# Get the datasets

## 1. Create target dataset staging folder
```bash
mkdir -p datasets/COWC_Points_512
```
## 2. Move ref_car.png from the repository root into the dataset folder
```bash
mv ref_car.png datasets/COWC_Points_512/ref_car.png
```

📂 Data Setup & Extraction
Download the COWC and VEDAI Raw data from the Google Drive Link and place the .zip / .tar files into the data/ directory. The raw data will be located under the data folder in the Google Drive
https://drive.google.com/drive/folders/1FTX76Ybf0PLiqdwyKsCvyi2WsF-ccoxY

Run these exact commands to unpack all dataset files automatically:
## Make and move into the data folder
```bash
mkdir -p data
cd data

# Uncompress COWC dataset
# On the Google Drive, the COWC.zip file is in the COWC_RAW folder
# See: https://drive.google.com/drive/folders/1VMFEMDJFNktTk5jbTBxdbuE9ii9rSwxf

unzip COWC.zip
```

# Uncompress VEDAI dataset

Grab VEDAI_RAW from Google Drive: https://drive.google.com/drive/folders/1VMFEMDJFNktTk5jbTBxdbuE9ii9rSwxf

```bash
mv VEDAI_RAW VEDAI
cd VEDAO
tar -xvf VEDAI_512_Annotations.tar
cat VEDAI_Vehicules512.tar.* > Vehicules512.tar
tar -xvf Vehicules512.tar

# Return back to the repository root directory
cd ../..
```
🏁 Quickstart Workflows
⚠️ Important: Always make sure you are at the repository root (label-refiner) and your environment is active (conda activate label-refiner) before running scripts. The scripts will automatically generate the required output directories (datasets/, datasets_refined/, and processed_datasets/).

# Option 0: Run on any dataset

Includes options for including a reference image, DINOv2 similarity scoare, box prompting, and snap. This is the main refinement path; it replaces the refinement steps of Options A and B. To score settings against true labels, see "Tuning on VEDAI" at the end of this section.

## 1. Set up data

Set up a data folder with /images and /labels in COWC format. The labels are generated using point level annotations and generic box size. You can use the python scripts/00pure_geographic_splitter.py for COWC data, or `scripts/VS_VEDAI_Separation_Conversion.py` for VEDAI (see "Tuning on VEDAI" below).

The input can be a flat folder or one with splits. The output copies the same layout.
```
COWC_mini_test/mini_dataset/       or     my_dataset/
├── images/   (*.png / *.jpg)             ├── train/{images,labels}/
└── labels/   (same names, .txt)          ├── val/{images,labels}/
                                          └── test/{images,labels}/
```

Labels are YOLO point labels, one car per line: `class cx cy w h` (normalized 0–1). The `w h` box is only used as a fallback when SAM's mask is rejected.

The COWC mini dataset used in the examples below (11 tiles, 65 cars, from the COWC test split) isn't in git, because folders named `*dataset*` are ignored. Rebuild it from the checked-in list after running `00pure_geographic_splitter.py`:
```bash
python scripts/make_cowc_subset.py    # reads COWC_mini_test/image_list.txt, writes COWC_mini_test/mini_dataset/
```

## 2. Generage refined YOLO OBB labels

```bash
python scripts/refine_labels.py --input COWC_mini_test/mini_dataset --output COWC_mini_test/mini_dataset_refined \
    --ref datasets/COWC_Points_512/ref_car.png
```
This takes about 3 s per image on CPU.

The reference car image (`--ref`) should be a tight crop of one car from the same kind of imagery. If it was enlarged (e.g. a 30 px car saved at 300 px), the script detects this and shrinks it back so the car is the same size as in the dataset, then segments the car with SAM. Without `--ref`, the script uses `<input>/ref_car.png` if it exists, otherwise it builds the reference from the labeled image with the most cars. The output folder contains:
- `images/`, `labels/`: copied images and refined YOLO OBB labels (`class x1 y1 x2 y2 x3 y3 x4 y4`)
- `refinement_report.csv`: one row per box, with how it was made (`source`), its similarity to the reference, and whether it was flagged
  - `sam`: SAM mask accepted
  - `corrected`: replaced by the geometric correction head (size and angle copied from a nearby car)
  - `fallback`: SAM mask rejected, the original placeholder box was kept
  - `similarity`: how car-like the label point looks compared with the reference; `background_similarity`: the image's median similarity; `contrast`: the difference
  - `low_similarity = 1`: contrast below `--sim-margin`, i.e. the point looks little more car-like than the background. Worth checking by eye (often cars cut off by the image edge or in shadow); it is a review hint, not a reliable bad-label detector
- `reference_mask.png`: the reference as SAM saw it, with the car(s) used for the PerSAM signature outlined in green. Check this first if the flags look wrong
- `data.yaml`: only for the train/val/test layout

Useful options:
| Option | Effect |
|---|---|
| `--car-length 30` | Typical car length in pixels (default 30, COWC). All pixel thresholds scale with it; use 20 for VEDAI 512 |
| `--ref PATH` | Reference car crop (default: `<input>/ref_car.png` if present) |
| `--ref-scale 0.1` | Override the automatic resize of the crop (e.g. if the car mask in `reference_mask.png` is the wrong size) |
| `--ref-mask PATH` | Use your own car mask for the crop instead of SAM's |
| `--ref-tile PATH` | Build the reference from this labeled image instead of a crop (default when no crop: the input image with the most labels) |
| `--sim-margin 0.10` | Flag boxes whose contrast is below this (default 0.10) |
| `--drop-low-sim` | Leave flagged boxes out of the label files (they stay in the report) |
| `--snap` | Move each prompt to the most car-like point within 8 px of the label (scaled by `--car-length`) |
| `--negative` | Add a negative prompt at the least car-like point in the image |
| `--box-prompt` | Prompt SAM with a box around each car (36 px at the default car length, i.e. 1.2 × `--car-length`) plus a ring of negative points just outside it (as in `VS_PerSAM_DINOv2.py`). Stops masks bleeding into neighbouring cars and shadows |
| `--box-half-size 18` | Set the box prompt half-width in pixels directly instead of scaling it from `--car-length` |
| `--features dino` | Use DINOv2 instead of SAM features for the similarity map (needs the DINOv2 files below) |

`--snap` and `--negative` are off by default because they reduced the number of accepted SAM masks on the COWC mini dataset. They may help for data with less accurate point labels.


Results on the mini dataset (65 cars; "sam" = box from an accepted SAM mask):
| Run | sam | corrected | fallback |
|---|---|---|---|
| default | 24 | 18 | 23 |
| `--box-prompt` | 54 | 2 | 9 |
| `--features dino --box-prompt --snap` (tile reference) | 55 | 3 | 7 |

With `--box-prompt`, many boxes are exactly the box size (36 px at the default car length), so long cars may be clipped. Check the plots.

**DINOv2 and the reference:** DINOv2 only works well with a reference *tile* (a real dataset image), not a hand-cropped car. On the mini dataset, the `ref_car.png` crop gave DINOv2 almost no car-vs-background contrast, because DINOv2 features depend on the car's surroundings. When using `--features dino`, leave out `--ref` (and make sure there's no `ref_car.png` in the input folder) or pass `--ref-tile`.

**Car size:** the pixel thresholds were tuned for COWC (cars about 30 px long). They scale with `--car-length`: the mask size filter (6–90 px), the correction head's valid-car range (15–42 px), the `--box-prompt` box, the SAM point spread, the `--snap` radius and the `--negative` exclusion radius. Aspect-ratio limits don't depend on scale and stay fixed. Set `--car-length` to the median car length of your data. The script prints the scale factor it is using.

### DINOv2 files (only for `--features dino`)
The script loads DINOv2 from local files, so no internet access is needed when it runs. These files are in `.gitignore`, so download them yourself:
1. Download the repository as a ZIP from https://github.com/facebookresearch/dinov2 (Code → Download ZIP) and unzip it to `models/dinov2-main/`
2. Download the ViT-S/14 weights (~85 MB) from https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth into `models/`

I've also put those files on Google Drive here: https://drive.google.com/drive/folders/1FTX76Ybf0PLiqdwyKsCvyi2WsF-ccoxY

## 3. Draw the boxes over the images
```bash
python data_exploration/plot_obb_labels.py \
    --images COWC_mini_test/mini_dataset_refined/images --labels COWC_mini_test/mini_dataset_refined/labels \
    --input-labels COWC_mini_test/mini_dataset/labels --out COWC_mini_test/mini_dataset_refined/plots
```
Colors: green = SAM mask accepted, cyan = correction head, orange = fallback, magenta X = low similarity (flagged), blue dot = snapped prompt, red dot = original point label.

## 4. Build a side-by-side comparison page
Pass one `--run NAME=FOLDER` per refined output folder (each needs `plots/` from step 4). They are shown left to right.
```bash
python data_exploration/compare_plots_html.py \
    --run "original=COWC_mini_test/mini_dataset_refined_original" \
    --run "default=COWC_mini_test/mini_dataset_refined" \
    --out COWC_mini_test/mini_compare.html
```

## 5. Optional (if on remote machine): View the page from a remote machine (no download needed)
On the remote machine, start a web server bound to localhost only, inside `tmux` so it keeps running after you log out:
```bash
tmux new -d -s plots 'cd ~/label-refiner && conda run -n label-refiner python -m http.server 8765 --bind 127.0.0.1'
```
It takes about 10 seconds to start. Stop it with `tmux kill-session -t plots`.

On your own computer, forward port 8765 over SSH:
- **Windows (PowerShell / OpenSSH)**: add to `%USERPROFILE%\.ssh\config` (no `.txt` extension):
  ```
  Host analysis.jpl.nasa.gov
      LocalForward 8765 localhost:8765
  ```
- **PuTTY**: Connection → SSH → Tunnels, Source port `8765`, Destination `localhost:8765`, click **Add**, then save the session.
- **Mac/Linux**: `ssh -N -L 8765:localhost:8765 <user>@<host>`

Then open http://localhost:8765/COWC_mini_test/mini_compare.html in your browser. The page reloads by itself whenever it is regenerated (step 5), so you can leave the tab open between runs.


## Tuning on VEDAI (truth labels, with a holdout set)

VEDAI has hand-drawn vehicle outlines, so refinement settings can be scored against them. Only car-sized classes are used: car, van and pickup (VEDAI classes 1, 9, 11; median box 21 × 9 px). Trucks, camping cars and tractors are larger and would need their own size settings.

**Holdout:** VEDAI's official `fold01test.txt` (121 images, 239 vehicles) is the holdout set and is never used for tuning. The tune split is the other 1,125 images (2,194 vehicles). Compare settings on the tune split, then check the final settings once on the holdout so the parameters aren't overfit.

1. **Convert** the raw data (in `data/VEDAI/`, see "Get the datasets") into center-point inputs and true OBB labels. Each point gets a 21 × 9 px placeholder box, used as the fallback when SAM's mask is rejected:
   ```bash
   python scripts/VS_VEDAI_Separation_Conversion.py
   ```
   This writes `datasets/vedai_512/vedai_color_centerpoint/` (input) and `datasets/vedai_512/vedai_color_obb/` (truth), plus the same for IR. Tuning uses the color images.

2. **Build the quick-check set** (12 tune images, 26 vehicles). The image list is checked in, so this rebuilds the same set:
   ```bash
   python scripts/make_vedai_subset.py --out VEDAI_mini_test --list VEDAI_mini_test/image_list.txt
   ```
   This creates `VEDAI_mini_test/mini_dataset/` (input), `truth/labels` and `truth/plots`. Use `--n-with N --n-empty M --seed S` instead of `--list` to sample a different set.

3. **Run the sweep** (8 settings, about 10 minutes on CPU), then summarize it:
   ```bash
   bash scripts/run_vedai_sweep.sh
   python scripts/summarize_vedai_sweep.py
   ```
   The sweep runs `refine_labels.py` with `--car-length 21` for each setting listed at the top of `scripts/run_vedai_sweep.sh` (default, box prompt, and box half-sizes 5–14 px; snap, the negative point and DINOv2 were dropped after they lowered IoU on the check set), scores each run and draws its plots. Finished runs are skipped, so an interrupted sweep can be restarted. The summary writes `VEDAI_mini_test/sweep_summary.txt` (one row per setting, sorted by mean IoU) and `VEDAI_mini_test/mini_compare.html` (the true boxes, then each run from best to worst; view it as in step 5 above). Set `PY=/path/to/python` if `python` isn't the label-refiner env.

4. **Overnight: the whole tune split.** Build a subset with every tune image and run the same sweep on it (about an hour per setting):
   ```bash
   python scripts/make_vedai_subset.py --out VEDAI_tune --all
   V=VEDAI_tune bash scripts/run_vedai_sweep.sh
   V=VEDAI_tune python scripts/summarize_vedai_sweep.py
   ```
   To keep it running after you log out, start it in tmux: `tmux new -d -s sweep 'cd ~/label-refiner && conda run -n label-refiner bash -c "V=VEDAI_tune bash scripts/run_vedai_sweep.sh" > VEDAI_tune_sweep.log 2>&1'`.

5. **Check the chosen settings on the holdout**, once:
   ```bash
   python scripts/refine_labels.py --input datasets/vedai_512/vedai_color_centerpoint \
       --output datasets_refined/vedai_color_final --car-length 21 <chosen options>
   python scripts/evaluate_updated_vedai_labels.py --pred datasets_refined/vedai_color_final --split holdout
   ```

**What the scores mean** (`scripts/evaluate_updated_vedai_labels.py`): each true vehicle is matched to one refined box by oriented IoU. The script reports mean and median IoU (unmatched vehicles count as 0), the share of vehicles at IoU ≥ 0.5 and ≥ 0.7, and median length, width and angle errors, overall and per `source` (sam / corrected / fallback). `--split tune|holdout|all` picks the images, `--truth` the truth labels, and `--fold N` another holdout fold. Results go to `evaluation_<split>_fold01.txt` and a per-vehicle `.csv` in the run folder.

**Baseline:** plain 21 × 9 px boxes centered on the points (no SAM) score a mean IoU of 0.47 on the tune split (39% of vehicles at IoU ≥ 0.5). A refinement setting should beat that.

# Option A: VEDAI Study Pipeline

> **Superseded by Option 0** (see "Tuning on VEDAI" above). Step 1 is still used to prepare the VEDAI data for Option 0. Steps 2–3 are kept for reference only: `refine_labels.py` now has this script's box prompt and DINOv2 features, and `VS_PerSAM_DINOv2.py` downloads DINOv2 from github.com, which fails on analysis.jpl.nasa.gov. `VS_evaluate_pipeline.py` counts trucks (class 2) as targets and has no holdout set; use `scripts/evaluate_updated_vedai_labels.py` instead.

## 1. Separate and convert raw dataset into working format
```bash
python scripts/VS_VEDAI_Separation_Conversion.py
```
## 2. Run SAM refinement with DINOv2 feature matching
Before running this script, you must crop a vehicle from the dataset to use as a reference photo. Save it as ref_car.png and place it in the root directory of the VEDAI centerpoint dataset. Make sure the crop is tight around the car so its full outline is clearly visible without cutting off any edges. PerSAM's single-shot feature extraction depends heavily on the quality of this reference crop. By default, the script processes the color (RGB) dataset.
```bash
python scripts/VS_PerSAM_DINOv2.py
```
## 3. Evaluate overall pipeline precision/recall metrics
```bash
python scripts/VS_evaluate_pipeline.py
```

# Option B: COWC Processing, PerSAM Refinement & Sensor Degradation
> **Partly superseded by Option 0.** Step 1 is still used to prepare COWC data for Option 0. Step 2's `01cowc_persam_generate_dataset.py` is the older refinement script (kept for now; use `scripts/refine_labels.py`). Steps 3–5 (sensor degradation, validation, packaging) have no Option 0 equivalent yet and are still current.

Follow this sequence to process, refine, degrade, and lock down the COWC dataset:

## 1. Tile and preprocess raw COWC aerial imagery into 512x512 patches
```bash
python scripts/00pure_geographic_splitter.py
```
## 2. Extract SAM feature signatures and generate PerSAM refined YOLO OBB labels
```bash
python scripts/01cowc_persam_generate_dataset.py
```
This older script has no CLI options and reads `datasets/` and the reference `datasets/COWC_Points_512/ref_car.png` (with an Otsu-thresholded mask) from hardcoded paths. For reference handling, DINOv2, box prompting and the report CSV, use `scripts/refine_labels.py` (see "Option 0" above).
## 3. Apply realistic sensor artifacts and resolution degradation
```bash
python scripts/02sensor_degradation.py
```
## 4. Validate dataset quality using the exploration tools
```bash
python data_exploration/val_yolo_dataset.py
```
## 5. Compress and lock finalized package for training repo export
zip -r dataset.zip path/to/data


🔍 Data Exploration Tools
Interactive inspection tools located in data_exploration/ allow you to inspect bounding box visual overlays, local crops, and format compliance:

# Run interactive dataset inspector
To use this file select the datasets_refined/COWC_512 folder. In order to change which split you are validating locate the val_yolo_dataset.py script and update the configuration. Use A and D to cycle through the images press M when an image does not meet the standards for training or validation. (The val split is most important here)
```bash
python data_exploration/val_yolo_dataset.py
```
👥 Contributors
* Bridgit Graddy — Lead contributor to dataset processing, label refinement architectures, and pipeline automation.
* Emily Dunkel
* Mike Burl
