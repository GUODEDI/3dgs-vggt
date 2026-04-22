# 3DGS-VGGT: Geometric Reliability-Guided 3D Gaussian Splatting

Improving 3D Gaussian Splatting (3DGS) training with a geometric reliability weight map (W_geo) derived from VGGT outputs.

## Core Contribution

We identify a conceptual confusion in the original VGGT + 3DGS integration pipeline: the original single weight map W was applied to both the photometric loss and the depth regularization loss, conflating two distinct concepts:

- **VGGT low-confidence regions** — a *data quality* problem (input-side)
- **3DGS hard-to-model regions** — a *model capacity* problem (output-side)

We address this by splitting the single weight map W into two semantically independent weight maps:

- **W_geo** (geometric reliability) → weights the **depth regularization loss** only
- **W_appear** (appearance modelability) → weights the **photometric loss** only (implemented, validation pending)

This report focuses on the design and experimental validation of **W_geo**.

## Method

**W_geo formula** (three-component weighted sum):

```
W_geo = 0.5 · (1 - σ) + 0.3 · V + 0.2 · C
```

| Component | Meaning | Source |
|-----------|---------|--------|
| `(1 - σ)` | Depth estimation reliability | VGGT DPT depth head output (`depth_conf`), normalized and inverted |
| `V` | Cross-frame visibility | Interpolated from VGGT sparse track visibility scores (~2560 points) |
| `C` | Multi-view geometric consistency | Back-projection / re-projection error from sparse tracks |

Minimum clipping at 0.1 to avoid fully masking any region.

**Integration:**
W_geo is applied **only** to the depth regularization loss, not the photometric loss, because:
- Depth loss compares against a VGGT **estimate** (may be unreliable → needs filtering)
- Photometric loss compares against the **ground-truth photo** (always reliable → no filtering needed)

## Results

Experiments on 3 LLFF scenes (7000 iterations, evaluated on training views with PSNR / SSIM / L1):

| Scene | Method | L1 ↓ | PSNR ↑ | SSIM ↑ |
|-------|--------|------|--------|--------|
| Kitchen | Baseline | 0.00610 | 40.34 | 0.9890 |
| Kitchen | **W_geo** | **0.00599** | **40.40** | 0.9884 |
| Fern | Baseline | 0.01586 | 29.13 | 0.9650 |
| Fern | **W_geo** | **0.01320** | **31.05** | **0.9716** |
| Flower | Baseline | 0.04659 | 21.25 | 0.7818 |
| Flower | **W_geo** | **0.04177** | **21.75** | **0.8117** |

**Improvement:**

| Scene | L1 | PSNR | SSIM |
|-------|-----|------|------|
| Kitchen | -1.8% | +0.06 dB | -0.06% |
| **Fern** | **-16.8%** | **+1.92 dB** | **+0.68%** |
| **Flower** | **-10.4%** | **+0.50 dB** | **+3.82%** |

Fern and Flower show consistent improvement across all three metrics and on every single training view (5/5).
Kitchen has limited headroom (already 40+ dB baseline) but maintains essentially equivalent performance.

## Usage

```bash
# 1. VGGT inference -> depth maps, camera params, track-based weights
python vggt/export_vggt_for_3dgs.py --scene_dir vggt/examples/<scene>

# 2. Convert VGGT output to 3DGS format (COLMAP + depth PNG + weight maps)
python vggt/convert_vggt_to_3dgs.py \
  --vggt_dir vggt/examples/<scene>/vggt \
  --output_dir vggt/examples/<scene>/3dgs_input

# 3. Train 3DGS with W_geo weighting on depth regularization
python gaussian-splatting-main/train.py \
  -s vggt/examples/<scene>/3dgs_input \
  -d depths --weights weights \
  -m output/<scene> \
  --iterations 7000 --disable_viewer
```

## Repository Structure

```
3dgs-vggt/
├── vggt/                          # VGGT code + integration scripts
│   ├── export_vggt_for_3dgs.py    # VGGT inference pipeline (modified)
│   ├── convert_vggt_to_3dgs.py    # VGGT -> 3DGS COLMAP converter
│   ├── compute_w_appear.py        # W_appear computation (future work)
│   └── vggt/                      # VGGT model source
└── gaussian-splatting-main/       # 3DGS code with weight-map support
    ├── train.py                   # Training with W_geo-weighted depth loss
    ├── scene/cameras.py           # Camera class (supports W_geo, W_appear)
    ├── scene/dataset_readers.py   # Weight map loading
    └── utils/loss_utils.py        # Weighted L1 / SSIM losses
```

## Environment

- Python 3.10
- PyTorch 2.4.0 + CUDA 11.8
- GPU: NVIDIA RTX 3060 (12 GB) tested
- VGGT model: VGGT-1B

Install:
```bash
conda create -n gaussian_splatting python=3.10 -y
conda activate gaussian_splatting
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu118
pip install tqdm plyfile opencv-python joblib scipy "numpy<2"
pip install --no-build-isolation gaussian-splatting-main/submodules/diff-gaussian-rasterization
pip install --no-build-isolation gaussian-splatting-main/submodules/simple-knn
pip install --no-build-isolation gaussian-splatting-main/submodules/fused-ssim
```

## Acknowledgments

- [VGGT](https://github.com/facebookresearch/vggt) — Meta AI
- [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) — Inria

## License

Research and educational use only. See submodule LICENSE files for third-party terms.
