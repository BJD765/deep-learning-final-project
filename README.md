# Deepfake Detection using Deep Learning
> **COMP6826001 - Deep Learning Final Project**  
> BINUS University | Semester Odd 2025/2026

## Overview

This project implements a **deepfake detection system** using deep learning to classify face images as **REAL** or **FAKE**. We use a ResNet18 model trained on the **FaceForensics++** dataset with YOLO-based face detection preprocessing.

### Live Demo
**Try it here:** [Hugging Face Spaces](https://huggingface.co/spaces/masp307/DefakeNet)

---

## Repository Structure

```
deep-learning-final-project/
├── notebooks/
│   ├── baseline.ipynb                # Main training pipeline (BEST MODEL)
│   ├── yolo_preprocessing.ipynb      # Face detection & cropping
│   └── archive/                      # Ablation experiments
│       ├── ablation11.ipynb
│       ├── ablation22.ipynb
│       └── ablation3.ipynb
├── models/
│   ├── BEST_MODEL.pth                # Best trained model weights
│
├── data/
│   └── README.md                     # Dataset information
├── internal_files/                   # Reference docs (gitignored)
├── .gitignore
├── LICENSE
└── README.md
```

---

## Methodology

### Dataset: FaceForensics++ (C23)

| Category | Methods | Videos |
|----------|---------|--------|
| **REAL** | Original | 1,000 |
| **FAKE** | Deepfakes, FaceSwap, Face2Face, NeuralTextures, FaceShifter, DeepFakeDetection | 6,000 |

### Pipeline

1. **Preprocessing**: YOLOv8-Face detection -> crop faces -> resize to 299x299
2. **Data Split**: 70% train / 15% val / 15% test (video-level stratification)
3. **Model**: ResNet18 (pretrained on ImageNet)
4. **Loss**: Focal Loss (gamma=2.0) for class imbalance
5. **Augmentation**: Random crop, flip, color jitter, Gaussian blur

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Learning Rate | 1e-4 |
| Batch Size | 64 |
| Epochs | 20 |
| Image Size | 224x224 |

---

## Results

| Metric | Score |
|--------|-------|
| **Accuracy** | 93.31% |
| **F1-Score** | 93.42% |


---

## Quick Start

### Environment Setup

```bash
# Create conda environment
conda create -n dl_env python=3.11
conda activate dl_env

# Install PyTorch with CUDA
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130

# Install dependencies
pip install numpy pandas scikit-learn matplotlib seaborn tqdm ultralytics albumentations
```

### Run Notebooks

1. **Preprocessing**: `notebooks/yolo_preprocessing.ipynb` - Extract face crops from videos
2. **Training**: `notebooks/baseline.ipynb` - Train the deepfake detector

---

## Team Members

| Name | Student ID | Contribution |
|------|------------|--------------|
| Jesslyn Trixie E | 2702260514 | AI Engineer and researcher |
| Brian Juniarta D | 2702279363 | AI engineer and researcher  |
| Mochammad Aqsa SP | 2702302744 | Full-stack developer and AI researcher|


---

## License

MIT License - see [LICENSE](LICENSE) for details.

---
