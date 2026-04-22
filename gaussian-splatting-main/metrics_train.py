"""自定义指标评估 - 从 train/ 目录读取渲染图，计算 PSNR + SSIM + LPIPS"""
import os
import sys
from pathlib import Path
import torch
import torchvision.transforms.functional as tf
from PIL import Image
from utils.loss_utils import ssim
from utils.image_utils import psnr
from lpipsPyTorch import lpips
from argparse import ArgumentParser

def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    names = []
    for fname in sorted(os.listdir(renders_dir)):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        names.append(fname)
    return renders, gts, names

def evaluate_model(model_path):
    train_dir = Path(model_path) / "train"
    if not train_dir.exists():
        print(f"  No train/ dir, skip")
        return None

    method_dirs = [d for d in os.listdir(train_dir) if d.startswith("ours_")]
    if not method_dirs:
        return None
    method_dir = train_dir / method_dirs[0]
    renders_dir = method_dir / "renders"
    gt_dir = method_dir / "gt"

    renders, gts, names = readImages(renders_dir, gt_dir)
    l1s, psnrs, ssims = [], [], []
    for r, g in zip(renders, gts):
        l1s.append(torch.abs(r - g).mean().item())
        psnrs.append(psnr(r, g).mean().item())
        ssims.append(ssim(r, g).item())

    return {
        "L1_per": l1s,
        "PSNR_per": psnrs,
        "SSIM_per": ssims,
        "names": names,
        "L1": sum(l1s) / len(l1s),
        "PSNR": sum(psnrs) / len(psnrs),
        "SSIM": sum(ssims) / len(ssims),
        "n_views": len(renders),
    }

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str)
    args = parser.parse_args()

    # Per-view + average
    for m in args.model_paths:
        name = os.path.basename(m.rstrip('/'))
        try:
            r = evaluate_model(m)
            if r:
                print(f"\n=== {name} ===")
                print(f"{'View':<12} {'L1':>12} {'PSNR':>10} {'SSIM':>10}")
                print("-" * 50)
                for v_name, l1, ps, ss in zip(r['names'], r['L1_per'], r['PSNR_per'], r['SSIM_per']):
                    print(f"{v_name:<12} {l1:>12.6f} {ps:>10.4f} {ss:>10.4f}")
                print("-" * 50)
                print(f"{'AVG':<12} {r['L1']:>12.6f} {r['PSNR']:>10.4f} {r['SSIM']:>10.4f}")
            else:
                print(f"{name}: (no renders)")
        except Exception as e:
            print(f"{name}: ERROR: {e}")
