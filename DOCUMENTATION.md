# Virtual Staining Benchmark: HAT Repository Documentation

## 1. Repository Profile
- **Model Name:** HAT (Hybrid Attention Transformer)
- **Task:** Image Super-Resolution / Restoration (Adapted for Virtual Staining)
- **Paired/Unpaired:** Paired (Uses `PairedImageDataset` from BasicSR)
- **Original Environment:** Python 3.7/3.8, PyTorch >= 1.7 (Avoid 1.8)
- **Current Environment:** Python 3.11, PyTorch 2.5.1+cu121 (RTX 3050 Ti 4GB VRAM)

## 2. Environment & System Setup
The project is configured for a Windows 11 system with an **NVIDIA GeForce RTX 3050 Ti Laptop GPU** (4GB VRAM limit).

### Core Stack Details:
- **PyTorch:** `2.5.1+cu121`
- **BasicSR:** `1.3.4.9` (Surgical fix applied to `degradations.py` for `rgb_to_grayscale` compatibility)
- **NumPy:** `1.26.4` (Pinning below 2.0 to avoid compatibility breaks)
- **Installation:** The repository is installed in editable mode using `python setup.py develop`.
## 3. Dataset Handling Protocol

### Model Expectations (BasicSR / HAT):
The HAT model uses the `PairedImageDataset` class. To work correctly, it requires:
- **Two Root Paths:** `dataroot_lq` (Low-Quality/Input) and `dataroot_gt` (Ground-Truth/Target).
- **Matching Filenames:** The model pairs images by filename. For example, if it finds `slide_01.png` in the input folder, it expects to find a corresponding `slide_01.png` in the target folder.
- **Pixel-Perfect Alignment:** Since this is a "paired" task, the images must be perfectly aligned (the same tissue structures must be at the same pixel coordinates).

### Our Dataset Mapping Logic:
We transformed the original BCI and MIST sample datasets to fit these "Input vs. Target" expectations:

#### **A. BCI Dataset (H&E to IHC)**
The original source provided `HE` and `IHC` folders with `train` and `test` subdirectories.
- **Mapping:** 
    - Source `HE` (Input) $\rightarrow$ `dataroot_lq` in the project.
    - Source `IHC` (Target) $\rightarrow$ `dataroot_gt` in the project.
- **Project Structure:**
    - `datasets/BCI/train_HE` (Input for training)
    - `datasets/BCI/train_IHC` (Target for training)
    - `datasets/BCI/test_HE` (Input for testing)
    - `datasets/BCI/test_IHC` (Target for testing)

#### **B. MIST Dataset (Unstained to Modality)**
The original source used a "Domain A" and "Domain B" convention inside a `TrainValAB` folder.
- **Mapping:** 
    - Domain `A` (Unstained Input) $\rightarrow$ `dataroot_lq`.
    - Domain `B` (Stained Target) $\rightarrow$ `dataroot_gt`.
- **Project Structure (Repeated for PR, ER, HER2, Ki67):**
    - `datasets/MIST/<Modality>/train_A` (Unstained training input)
    - `datasets/MIST/<Modality>/train_B` (Stained training target)
    - `datasets/MIST/<Modality>/test_A` (Unstained test input)
    - `datasets/MIST/<Modality>/test_B` (Stained test target)

---
## 4. Execution Guide & Commands
All commands use `--batch_size 1` and Tiling to stay within the **4GB VRAM** limit.

### BCI (H&E to IHC)
- **Train:** `python hat/train.py -opt options/train/smoke_test_BCI.yml`
- **Test:** `python hat/test.py -opt options/test/smoke_test_BCI_test.yml`

### MIST PR
- **Train:** `python hat/train.py -opt options/train/smoke_test_MIST_PR.yml`
- **Test:** `python hat/test.py -opt options/test/smoke_test_MIST_PR_test.yml`

### MIST Ki67
- **Train:** `python hat/train.py -opt options/train/smoke_test_MIST_Ki67.yml`
- **Test:** `python hat/test.py -opt options/test/smoke_test_MIST_Ki67_test.yml`

### MIST HER2
- **Train:** `python hat/train.py -opt options/train/smoke_test_MIST_HER2.yml`
- **Test:** `python hat/test.py -opt options/test/smoke_test_MIST_HER2_test.yml`

### MIST ER
- **Train:** `python hat/train.py -opt options/train/smoke_test_MIST_ER.yml`
- **Test:** `python hat/test.py -opt options/test/smoke_test_MIST_ER_test.yml`

## 5. Technical Behavior & Performance

### Tiling Mode ("tile n/64")
To process high-resolution tissue images on a 4GB GPU, the **Tiling Strategy** is used:
- **What:** The image is split into 64 small tiles (e.g., 128x128).
- **Process:** The GPU processes one tile at a time to save memory.
- **Progress Bar:** You will see the progress bar (1/64 to 64/64) **restart** for every image in the test set. If there are 5 test images, it will "loop" 5 times.

### Estimated Durations
- **Smoke Training (5 iters):** ~60 seconds per command.
- **Inference (Full Test Set):** ~45 seconds per command.
- **Full Modality Suite:** Running all modalities takes approximately **10-15 minutes**.

## 6. 🚨 PHASE 1 HACKS (MUST REVERT ON HPC)
To accommodate the **4GB VRAM limit**, the following architectural parameters were reduced:
1. **Window Size:** Reduced from 16 to 8.
2. **Embedding Dimension:** Reduced from 144 to 96.
3. **Depths:** Reduced from [6, 6, 6, 6, 6, 6] to [4, 4, 4, 4].
4. **Crop Size (gt_size):** Reduced to 128.
5. **Upsampling Scale:** Set to 1 (Identity) for same-resolution virtual staining.
