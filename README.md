Label Refiner & Small Object Detection Tools
This repository provides an automated pipeline for preparing, exploring, and refining small-object datasets—specifically targeting COWC variants, VEDAI, and other aerial imagery sources.

The primary objective is to generate clean, consistent YOLO-style Oriented Bounding Box (OBB) labels by leveraging the Segment Anything Model (SAM) and Personalized SAM (PerSAM) for bounding box refinement.

🌴 Directory Structure
Use this file tree to check your directory setup.

Plaintext
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
│   ├── 01cowc_persam_generate_dataset.py
│   ├── 02sensor_degradation.py
│   ├── VS_VEDAI_Separation_Conversion.py
│   ├── VS_PerSAM_DINOv2.py
│   └── VS_evaluate_pipeline.py
└── data_exploration/               <-- Inspection and validation tools
    └── val_yolo_dataset.py
🛠️ Complete Beginner's Setup Guide
If you have never used Conda or Terminal before, follow these step-by-step commands in order.

1. Install Miniforge3 (Conda Environment Manager)
If you do not have Conda installed, run the following commands in your terminal:

On macOS (using Homebrew):

Bash
brew install miniforge
conda init "$(basename "$SHELL")"
(After running conda init, close and reopen your terminal window).

2. Clone Repository & Setup Conda Environment
Copy and paste these commands into your terminal to clone the code and build an isolated environment with Python 3.10:

Bash
# 1. Clone the repository and enter the folder
git clone https://github.jpl.nasa.gov/science-team-algorithms/label-refiner.git
cd label-refiner

# 2. Create the Conda environment
conda create -n label-refiner python=3.10 -y

# 3. Activate the environment
conda activate label-refiner

# 4. Install all required dependencies
pip install -r requirements.txt
pip install git+https://github.com/facebookresearch/segment-anything.git
3. Setup Reference Image Artifact
Move the reference image (ref_car.png) into the target dataset directory where PerSAM expects to find it.

Bash
# 1. Create target dataset staging folder
mkdir -p datasets/COWC_Points_512

# 2. Move ref_car.png from the repository root into the dataset folder
mv ref_car.png datasets/COWC_Points_512/ref_car.png
📂 Data Setup & Extraction
Download the raw DATA folder from the Google Drive Link and place the .zip / .tar files into the data/ directory.

Run these exact commands to unpack all dataset files automatically:

Bash
# Move into the data folder
cd data

# Uncompress COWC dataset
unzip COWC.zip

# Uncompress VEDAI dataset
cd VEDAI
tar -xvf Annotations512.tar
cat Vehicules512.tar.* > Vehicules512.tar
tar -xvf Vehicules512.tar

# Return back to the repository root directory
cd ../..
🏁 Quickstart Workflows
⚠️ Important: Always make sure you are at the repository root (label-refiner) and your environment is active (conda activate label-refiner) before running scripts. The scripts will automatically generate the required output directories (datasets/, datasets_refined/, and processed_datasets/).

Option A: VEDAI Study Pipeline
Runs the end-to-end VEDAI SAM + DINOv2 relabeling and evaluation pipeline:

Bash
# 1. Separate and convert raw dataset into working format
python scripts/VS_VEDAI_Separation_Conversion.py

# 2. Run SAM refinement with DINOv2 feature matching
python scripts/VS_PerSAM_DINOv2.py

# 3. Evaluate overall pipeline precision/recall metrics
python scripts/VS_evaluate_pipeline.py
Option B: COWC Processing, PerSAM Refinement & Sensor Degradation
Follow this sequence to process, refine, degrade, and lock down the COWC dataset:

Bash
# 1. Tile and preprocess raw COWC aerial imagery into 512x512 patches
python scripts/00pure_geographic_splitter.py

# 2. Extract DINOv2 feature signatures and generate PerSAM refined YOLO OBB labels
python scripts/01cowc_persam_generate_dataset.py

# 3. Apply realistic sensor artifacts and resolution degradation
python scripts/02sensor_degradation.py

# 4. Validate dataset quality using the exploration tools
python data_exploration/val_yolo_dataset.py

# 5. Compress and lock finalized package for training repo export
zip -r dataset.zip path/to/data
🔍 Data Exploration Tools
Interactive inspection tools located in data_exploration/ allow you to inspect bounding box visual overlays, local crops, and format compliance:

Bash
# Run interactive dataset inspector
python data_exploration/val_yolo_dataset.py

👥 Contributors
Bridgit Graddy — Lead contributor to dataset processing, label refinement architectures, and pipeline automation.