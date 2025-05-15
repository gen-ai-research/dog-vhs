# VHSMarker: A High-Precision Annotation Tool for Canine Cardiac Keypoint Detection and VHS Estimation

<!-- ![Project Banner](static/assets/img/banner.png) <!-- Add your banner image if available 

## Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Dataset Setup](#dataset-setup)
  - [Annotation Tool](#annotation-tool)
  - [MambaVHS Model](#mambavhs-model)
- [Usage](#usage)
  - [Annotation Tool](#annotation-tool-1)
  - [Model Training](#model-training)
  - [Inference](#inference)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)

-->

## Overview

VHSMarker is an integrated solution for canine cardiac analysis, comprising:
- **CCK Dataset**: 21,465 annotated canine thoracic radiographs
- **VHSMarker Tool**: Web-based annotation interface
- **MambaVHS**: State-of-the-art deep learning model for VHS estimation

This project enables accurate Vertebral Heart Score (VHS) calculation through precise cardiac keypoint detection, supporting veterinary diagnostics and research.

## Key Features

✨ **Comprehensive Dataset**  
✔ 21,465 high-quality canine radiographs  
✔ Six annotated cardiac keypoints per image  
✔ Diverse breeds and clinical conditions  

🖥 **Intuitive Annotation Tool**  
✔ Real-time VHS calculation  
✔ Side-by-side ground truth vs prediction comparison  
✔ Export functionality for model training  

🧠 **Advanced Deep Learning Model**  
✔ Mamba architecture for long-range dependencies  
✔ Hybrid convolutional-sequence modeling  
✔ High accuracy VHS prediction  

## Installation

### Prerequisites
- Python 3.8+
- Node.js 14+
- npm or yarn
- CUDA-enabled GPU (recommended for training)

### Dataset Setup
1. Download the CCK dataset:
   ```bash
   git lfs install
   git clone https://huggingface.co/datasets/gen-ai-researcher/vhs_dogheart_db

### VHSMarker Web Annotation Tool
  ```bash
  cp -r vhs_dogheart_db/static/ static/
  cd vhs_marker_tool
  npm install
  pip install -r requirements.txt
  python app.py
```

### Mamba VHS Model
```bash
  cd mamba_vhs
  pip install -r requirements.txt
  python model.py

```
