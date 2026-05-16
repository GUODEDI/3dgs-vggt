"""自定义指标评估 - 从 train/ 和/或 test/ 目录读取渲染图，计算 L1 + PSNR + SSIM"""
import os
from pathlib import Path
import torch
import torchvision.transforms.functional as tf
from PIL import Image
from utils.loss_utils import ssim
from utils.image_utils import psnr
from argparse import ArgumentParser


def readImages(renders_dir, gt_dir):
    renders, gts, names = [], [], []
    for fname in sorted(os.listdir(renders_dir)):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        names.append(fname)
    return renders, gts, names


def evaluate_split(model_path, split):
    """评估单个 split（train / test）"""
    split_dir = Path(model_path) / split
    if not split_dir.exists():
        return None

    method_dirs = [d for d in os.listdir(split_dir) if d.startswith("ours_")]
    if not method_dirs:
        return None
    method_dir = split_dir / method_dirs[0]
    renders_dir = method_dir / "renders"
    gt_dir = method_dir / "gt"

    if not renders_dir.exists() or not gt_dir.exists():
        return None

    renders, gts, names = readImages(renders_dir, gt_dir)
    if len(renders) == 0:
        return None

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


def evaluate_model(model_path):
    """同时评估 train 和 test split"""
    return {
        "train": evaluate_split(model_path, "train"),
        "test": evaluate_split(model_path, "test"),
    }


def print_split_result(name, split_label, r):
    if r is None:
        return
    print(f"\n=== {name} [{split_label}] ===")
    print(f"{'View':<12} {'L1':>12} {'PSNR':>10} {'SSIM':>10}")
    print("-" * 50)
    for v, l1, ps, ss in zip(r['names'], r['L1_per'], r['PSNR_per'], r['SSIM_per']):
        print(f"{v:<12} {l1:>12.6f} {ps:>10.4f} {ss:>10.4f}")
    print("-" * 50)
    print(f"{'AVG':<12} {r['L1']:>12.6f} {r['PSNR']:>10.4f} {r['SSIM']:>10.4f}  (n={r['n_views']})")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str)
    parser.add_argument('--summary_only', action='store_true', help='只输出均值，不输出逐帧')
    args = parser.parse_args()

    # 汇总表
    summary = []  # rows: (name, split, L1, PSNR, SSIM, n)

    for m in args.model_paths:
        name = os.path.basename(m.rstrip('/'))
        try:
            results = evaluate_model(m)
            for split, r in results.items():
                if r is None:
                    continue
                if not args.summary_only:
                    print_split_result(name, split, r)
                summary.append((name, split, r['L1'], r['PSNR'], r['SSIM'], r['n_views']))
        except Exception as e:
            print(f"{name}: ERROR: {e}")

    # 汇总输出
    if summary:
        print("\n" + "=" * 90)
        print("Summary table")
        print("=" * 90)
        print(f"{'Model':<50} {'Split':<6} {'L1':>10} {'PSNR':>8} {'SSIM':>8} {'N':>3}")
        print("-" * 90)
        for name, split, l1, ps, ss, n in summary:
            print(f"{name:<50} {split:<6} {l1:>10.6f} {ps:>8.4f} {ss:>8.4f} {n:>3}")
