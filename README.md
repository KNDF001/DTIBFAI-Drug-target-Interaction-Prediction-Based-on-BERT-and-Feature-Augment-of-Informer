# DTIBFAI - Drug-Target Interaction Prediction via Biological Feature Augmented Interaction

⚠️ ​**Important Notice**  
**This project requires additional files that are not included in the repository.**  
Please download the following resources before proceeding:  
- Pre-trained models: [BioBERT](https://example.com/biobert-download) | [ChambERT](https://example.com/chamberts-download)
- Dataset files: [Full Dataset Package](https://example.com/dataset-download)

## Table of Contents
- [Introduction](#introduction)
- [Requirements](#requirements)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Pretrained Models](#pretrained-models)
- [Usage](#usage)
- [Citation](#citation)
- [License](#license)

## Introduction
DTIBFAI is a deep learning framework for drug-target interaction prediction, integrating:
- Biological feature augmentation (ECFP fingerprints & dipeptide composition)
- Pre-trained language models (BioBERT and ChambERT)
- Hybrid neural architecture for interaction modeling

## Requirements
- Python 3.8+
- PyTorch 1.12+
- Transformers 4.28+
- CUDA 11.6 (GPU recommended)
- Other dependencies:
  ```bash
  numpy pandas scikit-learn tqdm

## Installation
1. Clone repository:
   ```bash
   git clone https://github.com/yourusername/DTIBFAI.git
   cd DTIBFAI
Install dependencies:
bash
pip install -r requirements.txt
Data Preparation
Download dataset package and extract to:

DTIBFAI/
└── dataset/
    └── sBioSNAP/
        ├── train.csv
        ├── val.csv
        └── test.csv
        
File structure requirements:
SMILES sequences in SMILES column
Protein sequences in Target Sequence column
ECFP fingerprints stored as lists in ECFP column
Pretrained Models
Download and place models in project root:

DTIBFAI/
├── biobert-v1.2/  # From BioBERT download
└── chamberts/      # From ChambERT download

Usage
Training

python train.py \
  --batch-size 16 \
  --epochs 50 \
  --lr 1e-5
  
Evaluation

python test.py \
  --model-path ./saved_models/best_model.pth \
  --test-data ./dataset/sBioSNAP/test.csv

Training logs are saved in output.txt, containing:

ROC curve data
Precision-recall values
Optimal threshold calculations
Citation
If using this work, please cite:

@misc{dtibfai2024,
  title={DTIBFAI: Biological Feature Augmented Interaction for Drug-Target Prediction},
  author={Your Name},
  year={2024},
  publisher={GitHub},
  howpublished={\url{https://github.com/yourusername/DTIBFAI}}
}

License
This project is licensed under the MIT License - see LICENSE for details.

Note: For technical support or dataset access issues, please open an issue in the repository.

