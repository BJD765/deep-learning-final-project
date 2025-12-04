# FaceForensics++ C23 – Data Pipeline (Project Version)

Goal:  
Build a **binary deepfake detector** using the **FaceForensics++ C23 Kaggle subset**. We follow the spirit of the
original FaceForensics++ paper by (i) using **per-frame binary classification** with a CNN and (ii) handling the
REAL/FAKE class imbalance explicitly.

---

## 1. Define the task and labels

1. Use `FF++_Metadata.csv` (or its shuffled version) as the **master metadata**:
   - Columns: `File Path`, `Label`, `Frame Count`, `Width`, `Height`, `Codec`, `File Size(MB)`.
   - In our Kaggle subset: `Label` ∈ {`REAL`, `FAKE`}.

2. Label scheme (binary classification):
   - `REAL`  → frames from videos in the `original/` folder (genuine YouTube videos).
   - `FAKE`  → frames from videos in any manipulated folder:
     - `Deepfakes/`
     - `Face2Face/`
     - `FaceSwap/`
     - `NeuralTextures/`
     - `FaceShifter/`
     - `DeepFakeDetection/` (Google/Jigsaw subset)
   - For training we map labels to integer IDs: `REAL = 0`, `FAKE = 1`.

3. Define the **unit of training**:
   - Model input (baseline): **single full video frame** resized to 224×224 (no face crop required).
   - Output: **one binary label** (REAL vs FAKE).
   - Face detection + cropping can be added later as an **extension**, but is not required for the baseline.

---

## 2. Choose videos (full vs subsampled)

By default we **use all available videos** to follow the paper's idea that more data helps, and we handle class
imbalance with a **class-weighted loss** instead of discarding data.

1. From `FF++_Metadata.csv`:
    - Filter rows by `Label`:
       - `REAL` videos: 1000 originals.
       - `FAKE` videos: 6000 manipulations.
    - **Default (full) setting:**
       - Use **all 1000 REAL** and **all 6000 FAKE** videos.
       - Later, during training, compensate the 1:6 imbalance using **class weights** in the loss function
          (higher weight for REAL, lower for FAKE), similar to the original paper.

2. Optional subsampled experiment (for comparison / ablation):
    - We can also define a **balanced subset** at the video level:
       - Randomly select `N_real = 1000` REAL videos and `N_fake = 1000` FAKE videos (one fake per real).
    - This reduces training time and simplifies analysis, at the cost of not using all fake videos.
    - If we run this variant, we save a CSV like `selected_videos_balanced.csv` listing only the chosen videos.

3. In all cases, we store the final video list actually used for training in a CSV such as `selected_videos.csv`
    with columns like `File Path`, `Label`, `Frame Count`, so that our splits are **reproducible**.

---

## 3. Split selected videos into train/val/test

Splits must be **video‑disjoint** (no video appears in more than one split) so that frames from the same video never leak across splits.

1. On `selected_videos.csv` (or the full metadata if no subsampling is used):
   - Use a **stratified split by `Label` (REAL/FAKE)**:
     - Train: ~70%
     - Val: ~15%
     - Test: ~15%
   - This preserves the REAL/FAKE proportion in each split.

2. This gives three dataframes / CSVs:
   - `train_videos.csv`
   - `val_videos.csv`
   - `test_videos.csv`

3. Each row corresponds to **one video**.  
   We will now generate frames *only* from videos listed in these splits.

---

## 4. Frame sampling inside each video (approx. 1 fps)

We want **more frames per video**, but not all frames. Lecturer analogy: 16 fps → sample at 1 fps; for our case, assume ~25 fps as a rough average.

For each video in train/val/test:

1. Read `Frame Count` from metadata (call it `F`).
2. Choose an **approximate fps sampling step**:
   - Assume original fps ≈ 25.
   - Sampling at ~1 fps → **step ≈ 25 frames**.

3. Compute candidate frame indices:
   - Start at a small offset to avoid first frames being all black (e.g. `start = step`).
   - Indices: `start, start + step, start + 2*step, ...` until `index < F`.

4. Cap the **maximum frames per video**:
   - To avoid long videos dominating:
     - Example: `max_frames_per_video = 40`.
   - So you take:
     - `min(number_of_candidate_indices, max_frames_per_video)` frames.

5. Result per video:
   - Short video (e.g. 300 frames):
     - F ≈ 300, step = 25 → ~12 frames
   - Long video (e.g. 1200 frames):
     - F ≈ 1200, step = 25 → ~48 frames → capped at 40
   - Typical: ~10–40 frames per video.

This implements your lecturer’s “more frames, but still spaced in time” idea.

---

## 5. Frame extraction and resizing (per sampled frame)

For every selected frame index in each video:

1. Use OpenCV to:
   - Open the video.
   - Seek to the frame index.
   - Decode that frame (BGR image).

2. Convert to RGB if needed and **resize the entire frame** to a fixed size, e.g. **224×224** pixels.

3. Optionally apply:
   - Padding to keep aspect ratio before resizing, if desired.
   - Minor pre-normalization (but real normalization can be in the training transforms).

4. Associate this resized frame with:
   - The `Label` from the video metadata.
   - The split (train/val/test).
   - Optional: video ID, frame index for future analysis.

> Note: A later extension could replace this with **face detection + cropping**, but the baseline model will be trained on full resized frames.

---

## 6. Saving the preprocessed images (offline dataset)

We will build a **folder-structured** dataset for easy loading.

1. Define a root folder, e.g. `data_frames/`.

2. For each split (`train`, `val`, `test`) and each binary class label:
   - Create directories:

       ```
       data_frames/
       train/
             REAL/
             FAKE/
       val/
         ...
       test/
         ...
     ```

3. For each resized frame image:
   - Build a filename with:
     - Video base name + frame index, e.g.:
       - `01_02__meeting_serious__YVGY8LOK_f0123.png`
    - Save under corresponding folder based on the **binary label**, e.g.:

    ```
    data_frames/train/FAKE/01_02__meeting_serious__YVGY8LOK_f0123.png
    ```

4. Optionally also save a small CSV per split:
   - Columns: `image_path`, `label`, `video_path`, `frame_index`.
   - Useful for debugging and analysis.

---

## 7. Dataset and DataLoader design (for later training)

Once images are preprocessed and saved:

1. For training, use a simple **image classification dataset** interface:
    - Either:
       - Use `ImageFolder`-style logic (root/split/class/filename) with classes `REAL` and `FAKE`.
       - Or a custom `Dataset` that reads from your per-split CSV.

2. Define transforms:
   - **Train transforms**:
     - Random horizontal flip
     - Random crop or resize
     - Color jitter (light)
     - Normalize with ImageNet mean/std
   - **Val/Test transforms**:
     - Resize → center crop
     - Normalize

3. Prepare `DataLoader`s for train, val, test:
   - `batch_size`: e.g. 32–64 (depending on VRAM).
   - `shuffle=True` for train, `False` for val/test.
   - `num_workers`: 2–8 (depending on CPU).
   - For the **full, imbalanced setting**, we can optionally use a **WeightedRandomSampler** (PyTorch) or
     equivalent to draw batches with a more balanced mix of REAL and FAKE frames.

---

## 8. Model and training (high level, no code)

1. Pick a pretrained CNN backbone (e.g., ResNet18 / ResNet34 / ResNet50).
2. Replace the final classification layer to output **2 logits** (REAL vs FAKE).
3. Use:
    - Loss: cross-entropy.
       - In the **full-data setting** (1000 REAL vs 6000 FAKE), apply **class weights** in the loss (higher weight for
          REAL, lower for FAKE) to compensate for the imbalance, following the strategy described in the
          FaceForensics++ paper.
   - Optimizer: Adam or SGD with momentum.
   - Training schedule:
     - N epochs (e.g., 20–40)
     - Evaluate on val each epoch
     - Save best model by val accuracy.

4. After training:
   - Evaluate on **test split** using the saved best model.
   - Report:
     - Overall accuracy
     - Per-class accuracy
     - Confusion matrix
     - Example images (correct vs incorrect predictions).

---

## 9. Summary of key design choices

- **Videos per class**:
  - Random subset of N videos per class (e.g. 400).
- **Frames per video**:
  - Approx. 1 fps sampling → every ~25 frames,
  - Capped at `max_frames_per_video` (e.g. 40).
- **Splits**:
  - Train/val/test at *video level*, stratified by class.
- **Unit of training**:
  - Face crops at 224×224, one label per image.
- **Balance**:
  - Same N videos per class,
  - Similar frames per video logic for all classes.

This pipeline matches your lecturer’s intent:
- More temporal coverage per video,
- Fewer videos overall,
- Balanced across classes,
- Still fully feasible on your hardware.
