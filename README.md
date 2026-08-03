# Label Refiner & Small Object Detection Tools

This repository provides an automated pipeline for preparing, exploring, and refining small-object datasets—specifically targeting COWC variants and other aerial imagery sources. 

The primary objective is to generate clean, consistent YOLO-style labels by leveraging the **Segment Anything Model (SAM)** and **Personalized SAM (PerSAM)** for bounding box refinement. The repository is structured to handle dataset transformation, inspection, automated quality filtering, and experimental split locking within a reproducible, isolated environment.

---

🚀 Features
Test Subset & Preprocessing Scripts (00*): Utility scripts like 00make_cowc_test.py, 00make_dota_dataset.py, and 00pure_geographic_splitter.py extract lightweight samples and preprocess raw formats for rapid pipeline debugging without altering core processing flows.

VEDAI Study Pipeline (VS_*): End-to-end workflow designed for VEDAI dataset processing, including dataset separation/conversion, PerSAM feature extraction, failure analysis, and evaluation.

DINO and SAM Precision Refinement: Refinement scripts (such as 01cowc_persam_generate_dataset.py and VS_PerSAM_DINOv2.py) leverage DINOv2 feature matching, dynamic bounding box shrinking, and morphological erosion to tightly bound densely clustered vehicles at aerial resolutions.

Sensor Degradation: 02sensor_degradation.py applies realistic sensor artifacts and resolution loss to simulate non-ideal capture conditions.

Automated Quality Filtering: Cross-reference validation tools and failure detection scripts (VS_FindFailures.py) to inspect problematic imagery and labels from generated sets.


Dataset Exploration Tools: Interactive inspection utilities located in data_exploration/ provide bounding box visualizations, localized zooming, pipeline metrics, and DETR/YOLO format validation.

🛠️ Environment Setup
To avoid dependency conflicts and ensure all scripts execute correctly, build an isolated environment specifically for this repository.

1. Clone the Repository
Bash
git clone https://github.jpl.nasa.gov/science-team-algorithms/label-refiner.git
cd label-refiner
2. Create & Activate Environment
Using Conda (Recommended):

Bash
conda create -n label-refiner python=3.10 -y
conda activate label-refiner
Using venv:

Bash
python3 -m venv .env
source .env/bin/activate  # macOS/Linux
.env\Scripts\activate     # Windows
3. Install Dependencies
Bash
pip install -r requirements.txt
⚠️ Note: If utilizing a specific hardware acceleration backend (e.g., NVIDIA CUDA or Apple Silicon MPS), verify your PyTorch and Torchvision lines in requirements.txt align with your hardware before running the installation.

📦 Handling Large Files (Git LFS)
This repository utilizes Git Large File Storage (LFS) to manage large model weights, such as the SAM .pth checkpoints. If you clone the repository and notice that the model files are only a few bytes in size (containing text pointers instead of actual weights), pull the binary assets manually:

Bash
git lfs install
git lfs pull

🏁 Quickstart Workflows
Option A: VEDAI Study Pipeline
For the VEDAI dataset execution sequence, run the VS_* suite in the following order:

Bash
# 1. Separate and convert raw dataset into working format
python scripts/VS_VEDAI_Separation_Conversion.py

# 2. Run SAM refinement with DINOv2 feature matching
python scripts/VS_PerSAM_DINOv2.py

# 3. Analyze failure modes across processed imagery
python scripts/VS_FindFailures.py

# 4. Evaluate overall pipeline precision/recall metrics
python scripts/VS_evaluate_pipeline.py

Option B: COWC Processing & Degradation Pipeline
Follow these steps to process, refine, degrade, and lock down the standard COWC dataset:

1. Place Source Datasets
Ensure your raw imagery datasets are positioned inside the datasets/ root directory:

Plaintext
datasets/
    ├── cowc/
    ├── cowc_rgb_1m/
    └── cowc_rgb_05m/
2. Add Your SAM Checkpoint
Place your downloaded Segment Anything model weights inside the models/ directory:

Plaintext
models/
    └── sam_vit_b.pth
3. Generate Refined Labels
Execute the PerSAM pipeline to refine your target dataset layout:

Bash
python scripts/01cowc_persam_generate_dataset.py
Outputs will be generated to: datasets_refined/cowc/

4. Apply Sensor Degradation
Simulate sensor noise and degradation on the refined target dataset:

Bash
python scripts/02sensor_degradation.py
5. Purge Quality Blacklists
Validate the generated data using the exploration tools:

Bash
python data_exploration/val_yolo_dataset.py
If your inspection generated a needs_refinement/ folder containing images that failed validation.

6. Zip and Export
Compress the finalized, locked baseline package so it can be moved to your training repository:

Bash
zip -r 20260615_42_cowc_base.zip processed_datasets/20260615_42_cowc_base

👥 Contributors
Bridgit Graddy — Lead contributor to dataset processing, label refinement architectures, and pipeline automation development.