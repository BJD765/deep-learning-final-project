Here’s a report‑style dataset description you can drop into a `.md` or your report.

---

**Dataset Choice and Justification**

For this project we use a video deepfake dataset based on **FaceForensics++ (FF++)**, obtained from a public Kaggle mirror in its **C23 compressed** form. FaceForensics++ is a widely used benchmark dataset for facial manipulation detection, introduced by Rössler et al. (ICCV 2019). It was specifically designed to evaluate deep learning models on challenging, realistic face forgeries. Using this dataset is appropriate for our final project because:

- It is **standard and well‑cited** in the deepfake detection literature, which makes our results easier to compare with prior work.
- It provides **high‑quality, legally shared data** curated for research use.
- The Kaggle mirror we use is already **organized and balanced**, reducing preprocessing overhead and keeping the project feasible within time and hardware limits.
- It directly matches our problem: **binary deepfake detection from video frames**.

---

**Original FaceForensics++ Dataset (Conceptual Overview)**

The original FaceForensics++ dataset consists of **1000 real “original” videos** collected from YouTube, each containing a mostly frontal, trackable human face without heavy occlusions. These original videos are then manipulated using several automated face manipulation methods, creating realistic forgeries. Over time, the authors extended FF++ with:

- **Four core manipulation methods**:
  - **Deepfakes** – autoencoder‑based identity swap.
  - **Face2Face** – facial expression / mouth movement reenactment.
  - **FaceSwap** – traditional 3D face swapping pipeline.
  - **NeuralTextures** – GAN‑based neural rendering with learned textures.
- **FaceShifter** – a more recent, high‑fidelity face swap method robust to occlusions.
- **Deep Fake Detection Dataset** – an additional set of manipulated videos originally released by Google and Jigsaw, hosted within FF++.
- **Multiple compression levels**:
  - **C0** (uncompressed),
  - **C23** (H.264 compression at CRF 23),
  - **C40** (stronger compression, more artifacts).

In the full official distribution, these components together form a large and diverse dataset with multiple protocols (binary real vs fake, per‑manipulation multi‑class, different compressions). Access to the full dataset is controlled via an application form and download script.

---

**Kaggle C23 Variant Used in This Project**

Instead of downloading via the official script, we use a **Kaggle‑hosted mirror** of the **FaceForensics++ C23** subset. This variant focuses on a single compression level (C23) and provides a **clean, balanced subset** of the full data:

- Directory structure (top‑level folders under FaceForensics++_C23):
  - `original/`
  - `Deepfakes/`
  - `Face2Face/`
  - `FaceSwap/`
  - `NeuralTextures/`
  - `FaceShifter/`
  - `DeepFakeDetection/`
  - `csv/` (metadata)

Using the provided metadata file FF++_Metadata.csv, we verified:

- There are **7000 videos** described in the metadata:
  - `1000` labeled **REAL**
  - `6000` labeled **FAKE**
- Grouping by top‑level folder shows:

  - `original`           → 1000 videos, all labeled REAL  
  - `Deepfakes`          → 1000 videos, labeled FAKE  
  - `Face2Face`          → 1000 videos, labeled FAKE  
  - `FaceSwap`           → 1000 videos, labeled FAKE  
  - `NeuralTextures`     → 1000 videos, labeled FAKE  
  - `FaceShifter`        → 1000 videos, labeled FAKE  
  - `DeepFakeDetection`  → 1000 videos, labeled FAKE  

- For every row in FF++_Metadata.csv, the corresponding video file exists on disk (no missing files were found in our checks).

This indicates that the Kaggle version contains:

- The **1000 original YouTube videos**.
- A **balanced subset of 1000 videos** for each of six manipulation families, including the Deep Fake Detection subset.  
  (The original Google/Jigsaw dataset is larger than 1000 videos; here it has been downsampled to 1000 to match the other folders.)
- Only the **C23 compression level**; other compressions (C0, C40) are not included.

In practice, this Kaggle mirror is a **normalized, balanced C23 slice** of FaceForensics++ that is well‑suited for coursework‑scale experiments.

---

**Metadata Structure**

Each video is described by a row in FF++_Metadata.csv, which serves as the **master metadata table**. The key columns are:

- `File Path` – relative path to the video file, e.g.  
  `DeepFakeDetection/01_02__meeting_serious__YVGY8LOK.mp4`
- `Label` – binary label: `REAL` for originals, `FAKE` for all manipulations.
- `Frame Count` – number of frames in the video.
- `Width`, `Height` – spatial resolution (typically 1920×1080).
- `Codec` – video codec (H.264 for C23).
- `File Size(MB)` – approximate file size in megabytes.

This metadata allows us to:

- Distinguish between **real and fake** videos.
- Identify the **manipulation family** via the top‑level folder in `File Path`.
- Estimate how many frames can be sampled from each video.

---

**Task Definition: Binary Deepfake Detection**

Although FaceForensics++ supports both **binary** and **multi‑class** tasks, our Kaggle variant’s labels are already consolidated into:

- `REAL` – only videos from the `original/` folder.
- `FAKE` – all videos from:
  - `Deepfakes/`
  - `Face2Face/`
  - `FaceSwap/`
  - `NeuralTextures/`
  - `FaceShifter/`
  - `DeepFakeDetection/`

For this project, we define a **binary classification problem**:

- **Class 0 (Real):** frames sampled from videos labeled `REAL` (original folder).
- **Class 1 (Fake):** frames sampled from videos labeled `FAKE` (all manipulation folders combined).

This choice aligns with many real‑world deepfake detection scenarios, where the primary question is whether a given video/frame is authentic or manipulated, rather than which specific manipulation method was used. It also simplifies the model and evaluation, which is appropriate for a course final project.

As an optional extension, the dataset structure would also allow a **multi‑class** experiment by using the top‑level folder (`original`, `Deepfakes`, `Face2Face`, `FaceSwap`, `NeuralTextures`, `FaceShifter`, `DeepFakeDetection`) as the class label, but this is not our main focus.

---

**Why This Dataset and Variant Are Appropriate**

- **Relevance to the task:** FaceForensics++ is explicitly designed for deepfake detection, directly matching our chosen problem.
- **Balanced and manageable size:** The Kaggle C23 variant provides a **balanced** 7×1000 video set (1000 real, 6000 fake) which is large enough for meaningful experimentation but still manageable in terms of storage and compute, especially when we further subsample videos and frames.
- **Clean labeling:** The metadata already encodes a simple **REAL/FAKE** label scheme, reducing the risk of labeling errors and simplifying the data pipeline.
- **Realistic difficulty:** Using C23 compressed videos introduces realistic compression artifacts similar to those in online video platforms, making the detection task more practical than using only uncompressed footage.
- **Ethical and legal use:** FaceForensics++ is a research dataset released under specific terms of use; the Kaggle mirror we use is publicly available and widely adopted in academic projects. We only use the data for non‑commercial educational purposes and transparently document its origin and limitations.

---

If you’d like, I can next help you write a matching “Data Preparation and Pipeline” section that describes your frame sampling (1 fps, max frames per video), splitting, and preprocessing in the same report‑ready style.