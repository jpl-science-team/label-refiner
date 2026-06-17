# Label Refiner & Small Object Detection Tools

This repository provides an automated pipeline for preparing, exploring, and refining small-object datasets—specifically targeting COWC variants and other aerial imagery sources. 

The primary objective is to generate clean, consistent YOLO-style labels by leveraging the **Segment Anything Model (SAM)** and **Personalized SAM (PerSAM)** for bounding box refinement. The repository is structured to handle dataset transformation, inspection, automated quality filtering, and experimental split locking within a reproducible, isolated environment.


**Author: Bridgit Graddy, JPL Summer Intern, 2026** <br>
bgraddy@broncos.uncfsu.edu <br>>
Mentors: Emily Dunkel and Mike Burl

---

## 🚀 Features

* **Test Subset Generation:** `00create_test_subset.py` extracts a lightweight sample of each dataset (images + labels) for rapid pipeline debugging. All scripts starting with 00 are for testing/cleaning purposes and do not effect the current pipeline.
* **PerSAM Precision Refinement:** `01_cowc_persam_generate_dataset.py` leverages DINOv2 feature matching, dynamic bounding box shrinking, and morphological erosion to tightly bound densely clustered vehicles at 15m/pixel resolutions.
* **Automated Quality Filtering:** `00clean_refined_dir.py` cross-references a `needs_refinement` blacklist folder to instantly purge problematic imagery and labels from your generated sets.
* **Deterministic Split Locking:** `02lock_cowc_yolo_dataset.py` pools generated data, shuffles it with a fixed random seed, and locks in a clean **80% Train / 15% Val / 5% Test** YOLOv8-compliant distribution ready for deployment.
* **Dataset Exploration Tools:** Interactive inspection utilities located in `data_exploration/` provide bounding box visualizations and localized zooming.

---

## Large Files on Google Drive

The Science Team shared drive is located: https://drive.google.com/drive/folders/0AOVva14csqXNUk9PVA

Please contact Emily Dunkel for access to drive.


## 🛠️ Environment Setup

To avoid dependency conflicts and ensure all scripts execute correctly, build an isolated environment specifically for this repository.

### 1. Clone the Repository

```bash
git clone https://github.jpl.nasa.gov/science-team-algorithms/label-refiner.git
cd label-refiner
```

### 2. Create & Activate Environment

```bash
# using conda (recommended):
conda create -n label-refiner python=3.10 -y
conda activate label-refiner

# alternatively, using venv:
python3 -m venv .env
source .env/bin/activate  # macOS/Linux
.env\Scripts\activate     # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

⚠️ Note: If utilizing a specific hardware acceleration backend (e.g., NVIDIA CUDA or Apple Silicon MPS), verify your PyTorch and Torchvision lines in requirements.txt align with your hardware before running the installation.

📦 Handling Large Files (Git LFS)
This repository utilizes Git Large File Storage (LFS) to manage large model weights, such as the SAM .pth checkpoints. If you clone the repository and notice that the model files are only a few bytes in size (containing text pointers instead of actual weights), pull the binary assets manually.

A. Install Git LFS (If needed)

```bash
git lfs install
```

B. Pull Tracked Binary Assets

```bash
git lfs pull
```

### 🏁 Quickstart Workflow
Follow these steps to process a raw dataset from start to finish.

####  1. Place Source Datasets
Ensure your raw imagery datasets are positioned inside the datasets/ root directory:

```text
datasets/
    ├── cowc/
    ├── cowc_rgb_1m/
    └── cowc_rgb_05m/
```

#### 2. Add Your SAM Checkpoint
Place your downloaded Segment Anything model weights inside the models/ directory:

```text
models/
    └── sam_vit_b.pth
```

#### 3. Generate the Refined Labels
Execute the PerSAM pipeline to refine your target dataset layout:

```bash
python scripts/01cowc_persam_generate_dataset.py
Outputs will be generated to: datasets_refined/cowc/
```

#### 4. Purge Quality Blacklists

```bash
python data_exploration/val_yolo_dataset.py
```

If you have run the interactive dataset inspector tool and generated a needs_refinement/ folder containing images that failed validation, run the automated purge script to safely delete them from your newly generated data:

```bash
python scripts/00clean_refined_dir.py
```

#### 5. Lock and Freeze the Baseline Split
Pool all remaining clean imagery together, shuffle them deterministically, and lock them into a frozen 80/15/5 distribution complete with a relative-pathed data.yaml:

```bash
python scripts/02lock_cowc_yolo_dataset.py
```

Outputs will be written to: processed_datasets/20260615_42_cowc_base/ #update as follows <date_seed_dataset>

#### 6. Zip and Export
Compress the finalized, locked baseline package so it can be moved to your training repository:

```bash
zip -r 20260615_42_cowc_base.zip processed_datasets/20260615_42_cowc_base
```

👥 Contributors
Bridgit Graddy — Lead contributor to dataset processing, label refinement architectures, and pipeline automation development.
