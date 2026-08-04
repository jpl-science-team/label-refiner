# Label Refiner & Small Object Detection Tools

This repository provides an automated pipeline for preparing, exploring, and refining small-object datasets—specifically targeting COWC variants and other aerial imagery sources. 

The primary objective is to generate clean, consistent YOLO-style labels by leveraging the **Segment Anything Model (SAM)** and **Personalized SAM (PerSAM)** for bounding box refinement. The repository is structured to handle dataset transformation, inspection, automated quality filtering, and experimental split locking within a reproducible, isolated environment.

The Steps include to run this directory for complete beginers non coders, non conda users, non technical people to run 

---

🚀 Features
Test Subset & Preprocessing Scripts (00*): Utility scripts like 00make_cowc_test.py, 00make_dota_dataset.py, and 00pure_geographic_splitter.py extract lightweight samples and preprocess raw formats for rapid pipeline debugging without altering core processing flows.

VEDAI Study Pipeline (VS_*): End-to-end workflow designed for VEDAI dataset processing, including dataset separation/conversion, PerSAM feature extraction, failure analysis, and evaluation.

Sensor Degradation: 02sensor_degradation.py applies realistic sensor artifacts and resolution loss to simulate non-ideal capture conditions.

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
⚠️ Note: If utilizing a specific hardware acceleration backend (e.g., NVIDIA CUDA or Apple Silicon MPS), verify your PyTorch and Torchvision lines in requirements.txt align with your hardware before running the installation. Use LLMs if unable to install requirements for debugging. 

Data Structure
All data must be placed in the correct folder to ensure the scripts will run correctly.
To download data visit https://drive.google.com/drive/folders/1FTX76Ybf0PLiqdwyKsCvyi2WsF-ccoxY and download the raw DATA.

data/
    ├── VEDAI/
    ├── COWC/

In order for the data to be used it must be uncompressed.
1. Enter the directory where to data is stored

Bash
cd data

2. Unzip the COWC data and VEDAI tar files
Bash
unzip COWC.zip
cd VEDAI
tar -xvf Annotations512.tar
cat Vehicules512.tar.* > Vehicules512.tar
tar -xvf Vehicules512.tar

3. Make sure the uncompressed data in in the correct directory 

🏁 Quickstart Workflows
Run all scripts from the Repo root Label-Refiner/ ("cd .." to move up a directory)
Option A: VEDAI Study Pipeline
The VEDAI Study pipeline is for validating the DINO and SAM relabeling pipeline.
Before Running the SAM + DINO go into the dataset and make a ref_car.png and include it at the dataset root. Crop a single vehicle of your choosing to use as a reference. Crop as close to the car as possible making sure you can clearly see the border of the car in the complete image. 
For the VEDAI dataset execution sequence, run the VS_* suite in the following order:

Bash
# 1. Separate and convert raw dataset into working format
python scripts/VS_VEDAI_Separation_Conversion.py

# 2. Run SAM refinement with DINOv2 feature matching
python scripts/VS_PerSAM_DINOv2.py

# 3. Evaluate overall pipeline precision/recall metrics
python scripts/VS_evaluate_pipeline.py


Option B: COWC Processing & Degradation Pipeline
Follow these steps to process, refine, degrade, and lock down the standard COWC dataset:

1. Geographic COWC Splitter
Ensure your raw imagery datasets are positioned inside the data/ root directory. This is to be used to the raw COWC data only use step one if you are starting from the raw data.

Bash
Python scripts/00pure_geographic_splitter.py

3. Generate Refined Labels
Execute the perSAM pipeline to refine your target dataset layout:
Before Running the SAM + DINO go into the dataset and make a ref_car.png and include it at the dataset root. Crop a single vehicle of your choosing to use as a reference. Crop as close to the car as possible making sure you can clearly see the border of the car in the complete image. 

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


6. Zip and Export
Compress the finalized, locked baseline package so it can be moved to your training repository:

Bash
zip -r 20260615_42_cowc_base.zip processed_datasets/20260615_42_cowc_base

👥 Contributors
Bridgit Graddy — Lead contributor to dataset processing, label refinement architectures, and pipeline automation development.
