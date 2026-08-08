Label Refiner & Small Object Detection Tools
This repository provides an automated pipeline for preparing, exploring, and refining small-object datasets—specifically targeting COWC variants, VEDAI, and other aerial imagery sources.

The primary objective is to generate clean, consistent YOLO-style Oriented Bounding Box (OBB) labels by leveraging the Segment Anything Model (SAM) and Personalized SAM (PerSAM) for bounding box refinement.

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

│   ├── 01cowc_persam_generate_dataset.py

│   ├── 02sensor_degradation.py

│   ├── VS_VEDAI_Separation_Conversion.py

│   ├── VS_PerSAM_DINOv2.py

│   └── VS_evaluate_pipeline.py

└── data_exploration/               <-- Inspection and validation tools

    └── val_yolo_dataset.py

🛠️ Complete Beginner's Setup Guide

Note that these instructions are for running on a mac.

If you have never used Conda or Terminal before, follow these step-by-step commands in order.

1. Install Miniforge3 (Conda Environment Manager)
If you do not have Conda installed, run the following commands in your terminal:

On macOS (using Homebrew):


brew install miniforge
conda init "$(basename "$SHELL")"
(After running conda init, close and reopen your terminal window).

2. Clone Repository & Setup Conda Environment
Copy and paste these commands into your terminal to clone the code and build an isolated environment with Python 3.10:


# 1. Clone the repository and enter the folder
```bash
git clone https://github.jpl.nasa.gov/science-team-algorithms/label-refiner.git
cd label-refiner
```
# 2. Create the Conda environment
```bash
conda create -n label-refiner python=3.10 -y
```
# 3. Activate the environment
```bash
conda activate label-refiner
```
# 4. Install all required dependencies
```bash
pip install -r requirements.txt
```
# 5. Setup Reference Image Artifact
Move the reference image (ref_car.png) into the target dataset directory where PerSAM expects to find it.

# 1. Create target dataset staging folder
```bash
mkdir -p datasets/COWC_Points_512
```
# 2. Move ref_car.png from the repository root into the dataset folder
```bash
mv ref_car.png datasets/COWC_Points_512/ref_car.png
```

📂 Data Setup & Extraction
Download the COWC and VEDAI Raw data from the Google Drive Link and place the .zip / .tar files into the data/ directory. The raw data will be located under the data folder in the Google Drive
https://drive.google.com/drive/folders/1FTX76Ybf0PLiqdwyKsCvyi2WsF-ccoxY

Run these exact commands to unpack all dataset files automatically:
# Make and move into the data folder
```bash
mkdir -p data
cd data

# Uncompress COWC dataset
unzip COWC.zip

# Uncompress VEDAI dataset
mkdir -p VEDAI
mv *512* VEDAI/
cd VEDAI
tar -xvf Annotations512.tar
cat Vehicules512.tar.* > Vehicules512.tar
tar -xvf Vehicules512.tar

# Return back to the repository root directory
cd ../..
```
🏁 Quickstart Workflows
⚠️ Important: Always make sure you are at the repository root (label-refiner) and your environment is active (conda activate label-refiner) before running scripts. The scripts will automatically generate the required output directories (datasets/, datasets_refined/, and processed_datasets/).

Option A: VEDAI Study Pipeline
Runs the end-to-end VEDAI SAM + DINOv2 relabeling and evaluation pipeline:


# 1. Separate and convert raw dataset into working format
```bash
python scripts/VS_VEDAI_Separation_Conversion.py
```
# 2. Run SAM refinement with DINOv2 feature matching
Before running this script, you must crop a vehicle from the dataset to use as a reference photo. Save it as ref_car.png and place it in the root directory of the VEDAI centerpoint dataset. Make sure the crop is tight around the car so its full outline is clearly visible without cutting off any edges. PerSAM's single-shot feature extraction depends heavily on the quality of this reference crop. By default, the script processes the color (RGB) dataset.
```bash
python scripts/VS_PerSAM_DINOv2.py
```
# 3. Evaluate overall pipeline precision/recall metrics
```bash
python scripts/VS_evaluate_pipeline.py
```

Option B: COWC Processing, PerSAM Refinement & Sensor Degradation
Follow this sequence to process, refine, degrade, and lock down the COWC dataset:

# 1. Tile and preprocess raw COWC aerial imagery into 512x512 patches
```bash
python scripts/00pure_geographic_splitter.py
```
# 2. Extract DINOv2 feature signatures and generate PerSAM refined YOLO OBB labels
```bash
python scripts/01cowc_persam_generate_dataset.py
```
# 3. Apply realistic sensor artifacts and resolution degradation
```bash
python scripts/02sensor_degradation.py
```
# 4. Validate dataset quality using the exploration tools
```bash
python data_exploration/val_yolo_dataset.py
```
# 5. Compress and lock finalized package for training repo export
zip -r dataset.zip path/to/data

🔍 Data Exploration Tools
Interactive inspection tools located in data_exploration/ allow you to inspect bounding box visual overlays, local crops, and format compliance:

# Run interactive dataset inspector
To use this file select the datasets_refined/COWC_512 folder. In order to change which split you are validating locate the val_yolo_dataset.py script and update the configuration. Use A and D to cycle through the images press M when an image does not meet the standards for training or validation. (The val split is most important here)
```bash
python data_exploration/val_yolo_dataset.py
```
👥 Contributors
Bridgit Graddy — Lead contributor to dataset processing, label refinement architectures, and pipeline automation.
