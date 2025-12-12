# Data Directory

This folder contains the dataset files for the deepfake detection project.

## Structure

```
data/
├── raw/                    # Original dataset (do not commit)
│   └── FaceForensics++_C23/
├── processed/              # Preprocessed face crops
│   └── FFprocessed/
└── README.md               # This file
```

## Dataset: FaceForensics++

### Download Instructions

1. Request access from the official repository:
   - https://github.com/ondyari/FaceForensics

2. Download the C23 (HQ) version of the dataset

3. Extract to `data/raw/FaceForensics++_C23/`

### Contents

| Folder | Type | Videos | Description |
|--------|------|--------|-------------|
| `original/` | REAL | 1,000 | Original unmanipulated videos |
| `Deepfakes/` | FAKE | 1,000 | Deepfakes manipulation |
| `FaceSwap/` | FAKE | 1,000 | FaceSwap manipulation |
| `Face2Face/` | FAKE | 1,000 | Face2Face manipulation |
| `NeuralTextures/` | FAKE | 1,000 | Neural Textures manipulation |
| `FaceShifter/` | FAKE | 1,000 | FaceShifter manipulation |
| `DeepFakeDetection/` | FAKE | 1,000 | DeepFakeDetection manipulation |

### Preprocessing

Run the preprocessing notebook to extract face crops:

```bash
jupyter notebook notebooks/yolo_preprocessing.ipynb
```

This will:
1. Sample 20 frames per video
2. Detect faces using YOLOv8-Face
3. Crop and resize faces to 299×299
4. Save to `data/processed/FFprocessed/`

## Note

⚠️ **Do not commit data files to Git!**

The `.gitignore` file is configured to exclude:
- All video files (*.mp4, *.avi, *.mov)
- Raw and processed data directories
- Face crop folders

Store large files on cloud storage or use Git LFS if needed.
