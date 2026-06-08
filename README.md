
Label Refiner & Small Object Detection Tools
This repository provides a collection of tools for preparing, exploring, and refining small‑object datasets—specifically targeting the COWC variants and other aerial imagery sources.
The main objective is to generate clean, consistent YOLO‑style labels by leveraging the Segment Anything Model (SAM) for bounding box refinement.
The repo is structured to allow dataset transformation, inspection, refinement, and experimental model training, all within a reproducible, isolated environment.


Features
✔ Test Subset Generation
00create_test_subset.py extracts a small sample of each dataset (images + labels only) for rapid iteration.
✔ YOLO Label Refinement with SAM
01refine_labels_with_sam.py improves YOLO bounding boxes using mask‑derived bounding boxes from SAM, preserving originals if refinement fails.
✔ Dataset Exploration Tools
Quick utilities in data_exploration/ provide visualizations of YOLO datasets.
✔ End-to-End Dataset Workflow

Environment Setup
To avoid conflicts and ensure all scripts run as expected, create a clean environment specifically for this repo.
1. Create & Activate Environment
Using Conda (recommended):
conda create -n label-refiner python=3.10 -y
conda activate label-refiner

Using venv:
python3 -m venv .env
source .env/bin/activate  # macOS/Linux
.env\Scripts\activate     # Windows


2. Install Dependencies
Since you already maintain a requirements.txt, run:
pip install -r requirements.txt

(If using a specific PyTorch backend—CUDA or MPS—update the Torch lines in your requirements file accordingly.)

Quickstart
1. Place datasets in the repository
Datasets live under:
datasets/
    cowc/
    cowc_rgb_1m/
    cowc_rgb_05m/

2. Create a small test subset
python scripts/00create_test_subset.py

Output will appear under:
datasets/cowc_test_subset/

3. Add your SAM checkpoint
Place your .pth file inside:
models/

Example:
models/sam_vit_b.pth

4. Refine YOLO bounding boxes using SAM
python scripts/01refine_labels_with_sam.py

Results will be written to:
datasets/cowc_test_refined/



Contributor
Bridgit Graddy
Lead contributor to dataset processing, label refinement, and pipeline development.
