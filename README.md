Label Refiner & Small Object Detection Tools
This repository provides a collection of tools for preparing, exploring, and refining small-object datasets—specifically targeting the COWC variants and other aerial imagery sources.

The main objective is to generate clean, consistent YOLO-style labels by leveraging the Segment Anything Model (SAM) and Personalized SAM (PerSAM) for bounding box refinement. The repo is structured to allow dataset transformation, inspection, refinement, and experimental model training, all within a reproducible, isolated environment.

Features
Test Subset Generation: 00create_test_subset.py extracts a small sample of each dataset (images + labels only) for rapid iteration.

YOLO Label Refinement with SAM: 01refine_labels_with_sam.py improves YOLO bounding boxes using mask-derived bounding boxes from SAM, preserving originals if refinement fails.

PerSAM Precision Refinement: cowc_persam_refine_obb.py uses DINOv2 feature matching, dynamic bounding box shrinking, and morphological erosion to tightly bound densely clustered vehicles at 15m/pixel resolutions.

Dataset Exploration Tools: Quick utilities in data_exploration/ provide visualizations of YOLO datasets.

Environment Setup
To avoid conflicts and ensure all scripts run as expected, create a clean environment specifically for this repo.

1. Clone the Repository
First, bring the code down to your local machine:
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
Since you already maintain a requirements.txt, run:

Bash
pip install -r requirements.txt
(Note: If using a specific PyTorch backend—CUDA or MPS—update the Torch lines in your requirements file accordingly before running this command).

Handling Large Files (Git LFS)
This repository uses Git Large File Storage (LFS) to manage large model weights (like the SAM .pth checkpoints). If you clone the repository and your model files are only a few bytes in size (containing text pointers instead of the actual weights), you need to pull the LFS files.

1. Install Git LFS
If you haven't already, install Git LFS on your machine:

Bash
git lfs install
2. Pull the Tracked Files
To download the actual heavy files stored in LFS for this repository, run:

Bash
git lfs pull


Quickstart Workflow
1. Place Datasets
Datasets should live inside the datasets/ directory:

Plaintext
datasets/
    ├── cowc/
    ├── cowc_rgb_1m/
    └── cowc_rgb_05m/
2. Create a Small Test Subset
Extract a lightweight test set to rapidly test your pipeline:

Bash
python scripts/00create_test_subset.py
Output will appear under: datasets/cowc_test_subset/

3. Add Your SAM Checkpoint
Place your downloaded model weights inside the models/ directory:
Plaintext
models/sam_vit_b.pth

4. Run PerSAM Bounding Box Refinement
To use the advanced Personalized SAM script for tightly packed vehicles, ensure you have your visual templates (ref_car.png and ref_mask.png) placed in your dataset folder. 

Then run:
Bash
python scripts/cowc_rgb_1m_persam_refine_obb.py
Results will be written to: datasets/cowc_test_refined/

(If you just want standard SAM refinement, you can still run python scripts/01refine_labels_with_sam.py)

5. Explore the Dataset
Visualize the original or refined YOLO bounding boxes drawn directly onto the images to verify your results:

Bash
python data_exploration/explore_test_dataset.py

Contributors
Bridgit Graddy
Lead contributor to dataset processing, label refinement, and pipeline development.