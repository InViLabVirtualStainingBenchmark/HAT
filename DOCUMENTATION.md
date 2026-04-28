Hat documentation · MD
Copy

# HAT Virtual Staining — Repository Documentation
**Project:** Virtual Staining Benchmark | KdG Hogeschool | April 2026  
**Status:** Inference + Smoke Training confirmed working on BCI and MIST datasets
 
---
 
## 1. Repository Profile
 
| Field | Value |
|---|---|
| Model Name | HAT (Hybrid Attention Transformer) |
| Original Task | Image Super-Resolution (ImageNet) |
| Adapted Task | Virtual Staining (Image-to-Image Translation) |
| Pairing Mode | Paired — `PairedImageDataset` (BasicSR) |
| Scale | 1 (same resolution in/out — no upsampling) |
| Framework | BasicSR 1.4.2 |
| Repo | InViLab fork of official HAT repository |
 
---
 
## 2. Environment Setup
 
### Local Machine (Windows 11 — CPU only)
 
| Component | Version |
|---|---|
| Python | 3.9 (venv via `hat_env/`) |
| PyTorch | 2.1.2+cpu |
| TorchVision | 0.16.2+cpu |
| BasicSR | 1.4.2 |
| einops | installed |
| num_gpu | 0 in all configs |
| num_worker_per_gpu | 0 (Windows multiprocessing fix) |
 
### Activation Command
```powershell
cd C:\Users\edobo\IdeaProjects\HAT
hat_env\Scripts\Activate.ps1
```
 
### HPC (Future — not yet configured)
> When moving to HPC: change `num_gpu` to 1, `num_worker_per_gpu` to 4+, use CUDA PyTorch build.  
> The same configs work — only these fields need updating.  
> Datasets must be transferred separately via rsync (excluded from git).
 
---
 
## 3. Critical Architecture Fix: scale=1 Support
 
### Problem
HAT was designed exclusively for super-resolution (`upscale=4`). When `upsampler=''` and `scale=1`, the original `forward()` function **skipped all learned layers entirely** — returning the input unchanged with no gradient graph attached. This caused a `RuntimeError` on the first `.backward()` call during training.
 
### Root Cause in `hat/archs/hat_arch.py`
```python
def forward(self, x):
    x = (x - self.mean) * self.img_range
 
    if self.upsampler == 'pixelshuffle':   # only branch that ran the network
        x = self.conv_first(x)
        x = self.conv_after_body(self.forward_features(x)) + x
        x = self.conv_before_upsample(x)
        x = self.conv_last(self.upsample(x))
 
    # upsampler='' fell through here — no layers touched, no gradient!
    x = x / self.img_range + self.mean
    return x
```
 
### Fix Applied — Two Edits
 
**Edit 1 — `__init__`: Add `conv_last` for the scale=1 case (~line 869)**
```python
if self.upsampler == 'pixelshuffle':
    self.conv_before_upsample = nn.Sequential(...)
    self.upsample = Upsample(upscale, num_feat)
    self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
else:  # ADDED
    # For image-to-image translation (scale=1, virtual staining)
    self.conv_last = nn.Conv2d(embed_dim, num_out_ch, 3, 1, 1)
```
 
**Edit 2 — `forward()`: Add else branch (~line 975)**
```python
if self.upsampler == 'pixelshuffle':
    x = self.conv_first(x)
    x = self.conv_after_body(self.forward_features(x)) + x
    x = self.conv_before_upsample(x)
    x = self.conv_last(self.upsample(x))
else:  # ADDED
    # For image-to-image translation (scale=1, virtual staining)
    x = self.conv_first(x)
    x = self.conv_after_body(self.forward_features(x)) + x
    x = self.conv_last(x)  # no upsample — same resolution out
```
 
**Why this works:** With the `else` branch, all learned layers run for `scale=1`. The gradient graph is intact → `.backward()` works → training proceeds. `conv_last` maps `embed_dim` (96) → 3 channels (RGB output).
 
### Other Training Bugs Fixed
 
| Bug | Fix |
|---|---|
| `ema_decay: 0.999` | Set to `0` — EMA caused gradient issues training from scratch on CPU |
| `num_worker_per_gpu: 1` | Set to `0` — Windows multiprocessing deadlocks with workers > 0 |
| `num_gpu: 1` | Set to `0` — no CUDA available locally |
| `upsampler: 'pixelshuffle'` | Set to `''` — virtual staining is scale=1, no upsampling needed |
 
---
 
## 4. Dataset Handling
 
### How PairedImageDataset Works
BasicSR's `PairedImageDataset` pairs images strictly **by filename**. If `test_HE` contains `00001.png`, it expects `test_IHC` to also contain `00001.png`. Any mismatch in count or filename causes an `AssertionError` at startup.
 
---
 
### A. BCI Dataset — H&E to IHC
 
| Field | Value |
|---|---|
| Input (lq) | HE — Hematoxylin & Eosin stained |
| Target (gt) | IHC — Immunohistochemistry stained |
| Train pairs | 920 matched images (after deduplication) |
| Test pairs | 11 (borrowed from train set — smoke test only) |
| Issue | Source download was part 2 of 3 — HE and IHC had different counts |
| Fix | Intersection-based cleanup: kept only filenames present in both folders |
 
**Folder structure:**
```
datasets/BCI/
    train_HE/     ← dataroot_lq (training input)
    train_IHC/    ← dataroot_gt (training target)
    test_HE/      ← dataroot_lq (test input)
    test_IHC/     ← dataroot_gt (test target)
```
 
---
 
### B. MIST Dataset — Unstained to Modality
 
| Field | Value |
|---|---|
| Modalities | ER, HER2, Ki67, PR |
| Input (A) | Unstained tissue slide |
| Target (B) | Stained slide (specific protein marker) |
| Train pairs | 5 per modality (sample only) |
| Test pairs | 5 per modality |
 
**Folder structure:**
```
datasets/MIST/
    HER2/
        train_A/  ← dataroot_lq
        train_B/  ← dataroot_gt
        test_A/   ← dataroot_lq
        test_B/   ← dataroot_gt
    ER/    (same structure)
    Ki67/  (same structure)
    PR/    (same structure)
```
 
---
 
## 5. Configuration Files
 
### Network Architecture Parameters (Local / Smoke)
 
> ⚠️ These are **REDUCED** parameters for CPU-only local testing. Revert on HPC (see Section 8).
 
| Parameter | Local Value | Full Value (HPC) |
|---|---|---|
| `window_size` | 8 | 16 |
| `embed_dim` | 96 | 180 |
| `depths` | [4,4,4,4] | [6,6,6,6,6,6] |
| `num_heads` | [6,6,6,6] | [6,6,6,6,6,6] |
| `compress_ratio` | 24 | 3 |
| `squeeze_factor` | 24 | 30 |
| `gt_size` | 128 | 256 |
| `upsampler` | `''` | `''` (stays empty for virtual staining) |
 
### Training Config — Key Fields
 
| Field | Value |
|---|---|
| `ema_decay` | 0 |
| `num_gpu` | 0 |
| `num_worker_per_gpu` | 0 |
| `batch_size_per_gpu` | 1 |
| Optimizer | Adam, lr=2e-4, betas=[0.9, 0.99] |
| Loss | L1Loss |
| Scheduler | MultiStepLR |
| `total_iter` | 10 (smoke) / 920 (BCI 1 epoch) |
| `val_freq` | matches `total_iter` |
| `save_img` | false |
| `print_freq` | 5 |
 
### All Config Files
 
| File | Location | Purpose |
|---|---|---|
| `smoke_test_BCI.yml` | `options/train/` | BCI training |
| `HAT_BCI_test.yml` | `options/test/` | BCI inference |
| `smoke_test_MIST_HER2.yml` | `options/train/` | MIST HER2 training |
| `smoke_test_MIST_ER.yml` | `options/train/` | MIST ER training |
| `smoke_test_MIST_Ki67.yml` | `options/train/` | MIST Ki67 training |
| `smoke_test_MIST_PR.yml` | `options/train/` | MIST PR training |
| `HAT_MIST_HER2_test.yml` | `options/test/` | MIST HER2 inference |
 
---
 
## 6. Execution Commands
 
### Activate Environment
```powershell
cd C:\Users\edobo\IdeaProjects\HAT
hat_env\Scripts\Activate.ps1
```
 
### BCI
```powershell
# Inference (pretrained model — baseline)
python hat/test.py -opt options/test/HAT_BCI_test.yml
 
# Smoke training (10 iters)
python hat/train.py -opt options/train/smoke_test_BCI.yml
```
 
### MIST
```powershell
# HER2
python hat/test.py  -opt options/test/HAT_MIST_HER2_test.yml
python hat/train.py -opt options/train/smoke_test_MIST_HER2.yml
 
# ER
python hat/train.py -opt options/train/smoke_test_MIST_ER.yml
 
# Ki67
python hat/train.py -opt options/train/smoke_test_MIST_Ki67.yml
 
# PR
python hat/train.py -opt options/train/smoke_test_MIST_PR.yml
```
 
### Push to Git
```powershell
git add hat/archs/hat_arch.py options/train/ options/test/ .gitignore
git commit -m "fix: scale=1 support for virtual staining + BCI/MIST configs"
git push origin main
```
 
---
 
## 7. Smoke Test Results
 
### Inference — Pretrained ImageNet Model (Untrained Baseline)
 
> These numbers are expected to be poor — the model has never seen pathology images. They are the baseline before any training.
 
| Dataset | PSNR | SSIM |
|---|---|---|
| BCI (11 test images) | 14.89 dB | 0.4014 |
| MIST HER2 | confirmed working | — |
 
### Training — Loss (BCI Smoke)
 
| Checkpoint | Loss |
|---|---|
| Iter 50 | L1 = 0.1382 — model is learning |
| Expected end of epoch (iter 920) | ~0.05–0.08 range |
 
---
 
## 8. HPC Migration Checklist
 
Make these changes to **all config files** before running on HPC:
 
| Local Setting | HPC Setting |
|---|---|
| `num_gpu: 0` | `num_gpu: 1` |
| `num_worker_per_gpu: 0` | `num_worker_per_gpu: 4` |
| `window_size: 8` | `window_size: 16` |
| `embed_dim: 96` | `embed_dim: 180` |
| `depths: [4,4,4,4]` | `depths: [6,6,6,6,6,6]` |
| `num_heads: [6,6,6,6]` | `num_heads: [6,6,6,6,6,6]` |
| `compress_ratio: 24` | `compress_ratio: 3` |
| `squeeze_factor: 24` | `squeeze_factor: 30` |
| `gt_size: 128` | `gt_size: 256` |
| `batch_size_per_gpu: 1` | `batch_size_per_gpu: 4` |
| `total_iter: 10` | full epoch count per dataset |
 
### Dataset Transfer
```bash
rsync -avz datasets/BCI/  user@hpc:/path/to/HAT/datasets/BCI/
rsync -avz datasets/MIST/ user@hpc:/path/to/HAT/datasets/MIST/
rsync -avz experiments/pretrained_models/ user@hpc:/path/to/HAT/experiments/pretrained_models/
```
 
---
 
## 9. Next Steps
 
| Step | Task |
|---|---|
| 1 | Push HAT repo — `hat_arch.py` fix + all configs + `.gitignore` |
| 2 | Clone and set up NAFNet repository |
| 3 | Apply same process: fix scale=1, create BCI + MIST configs, smoke test |
| 4 | Once both models confirmed locally → set up HPC environment |
| 5 | Get full BCI dataset (parts 1 and 3 missing) |
| 6 | Run full training on HPC |
| 7 | Compare PSNR/SSIM: HAT vs NAFNet → benchmark table |