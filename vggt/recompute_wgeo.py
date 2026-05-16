"""
仅重算 W_geo 权重图（不重跑整个 VGGT 推理）。
输入：已有的 vggt/ 目录（含 sigma.npy / vis.npy / consistency 推算用的轨迹）。
输出：覆盖 frame_XXX_W.npy 文件，并更新 wgeo_config.json。

用途：W_geo 系数消融实验的轻量版。
"""

import argparse
import json
import os
from pathlib import Path
import numpy as np


def percentile_normalize(data, p_low=5, p_high=95):
    p_low_val = np.percentile(data, p_low)
    p_high_val = np.percentile(data, p_high)
    if p_high_val > p_low_val:
        normalized = (data - p_low_val) / (p_high_val - p_low_val)
        return np.clip(normalized, 0.0, 1.0)
    return np.ones_like(data)


def compute_weight_map(sr, vr, cr, w_sigma, w_vis, w_cons, strategy, min_weight):
    """与 export_vggt_for_3dgs.py 中保持一致"""
    # 系数全 0 时退化为均匀权重（避免除零）
    if cr is None:
        if w_sigma + w_vis < 1e-8:
            return np.full_like(sr, max(min_weight, 0.0))
        if strategy == "multiplicative":
            W = (sr ** w_sigma) * (vr ** w_vis)
        else:
            total = w_sigma + w_vis
            W = (w_sigma / total) * sr + (w_vis / total) * vr
    else:
        if w_sigma + w_vis + w_cons < 1e-8:
            return np.full_like(sr, max(min_weight, 0.0))
        if strategy == "multiplicative":
            W = (sr ** w_sigma) * (vr ** w_vis) * (cr ** w_cons)
        else:
            total = w_sigma + w_vis + w_cons
            W = ((w_sigma / total) * sr
                 + (w_vis / total) * vr
                 + (w_cons / total) * cr)
    W = np.maximum(W, min_weight)
    W = np.clip(W, 0.0, 1.0)
    return W


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vggt_dir", type=str, required=True,
                        help="VGGT 推理输出目录（含 frame_XXX_sigma.npy, frame_XXX_vis.npy 等）")
    parser.add_argument("--w_sigma", type=float, default=0.5)
    parser.add_argument("--w_vis", type=float, default=0.3)
    parser.add_argument("--w_cons", type=float, default=0.2)
    parser.add_argument("--w_strategy", type=str, default="weighted_avg",
                        choices=["multiplicative", "weighted_avg"])
    parser.add_argument("--w_min", type=float, default=0.1)
    args = parser.parse_args()

    vggt_dir = Path(args.vggt_dir)
    if not vggt_dir.exists():
        raise FileNotFoundError(f"vggt_dir 不存在: {vggt_dir}")

    # 找出所有帧
    sigma_files = sorted(vggt_dir.glob("frame_*_sigma.npy"))
    if not sigma_files:
        raise ValueError(f"在 {vggt_dir} 中找不到 frame_XXX_sigma.npy")

    print(f"找到 {len(sigma_files)} 帧，开始重算 W_geo...")
    print(f"系数：w_sigma={args.w_sigma}, w_vis={args.w_vis}, w_cons={args.w_cons}, "
          f"strategy={args.w_strategy}, min={args.w_min}")

    for sigma_path in sigma_files:
        stem = sigma_path.stem.replace("_sigma", "")  # frame_000_sigma -> frame_000
        vis_path = vggt_dir / f"{stem}_vis.npy"
        cons_path = vggt_dir / f"{stem}_cons.npy"

        sigma = np.load(sigma_path)
        sr = 1.0 - sigma  # 不确定度反向 = 可靠性

        if vis_path.exists():
            vis = np.load(vis_path)
            vr = percentile_normalize(vis)
        else:
            print(f"  [WARN] {vis_path.name} 不存在，跳过 vis")
            vr = np.ones_like(sigma)

        # 一致性：如果存在 _cons.npy 文件就加载，否则降级
        cr = None
        if cons_path.exists():
            cons = np.load(cons_path)
            cr = percentile_normalize(cons)
        else:
            print(f"  [INFO] {cons_path.name} 不存在，使用降级模式（仅 sigma+vis）")

        W = compute_weight_map(sr, vr, cr,
                                args.w_sigma, args.w_vis, args.w_cons,
                                args.w_strategy, args.w_min)

        out_path = vggt_dir / f"{stem}_W.npy"
        np.save(out_path, W.astype(np.float32))
        print(f"  {stem}: mean={W.mean():.3f}, <0.2={((W<0.2).sum()/W.size*100):.1f}%")

    # 更新 wgeo_config.json
    config = {
        "w_sigma": args.w_sigma,
        "w_vis": args.w_vis,
        "w_cons": args.w_cons,
        "strategy": args.w_strategy,
        "min_weight": args.w_min,
        "note": "Recomputed with recompute_wgeo.py (consistency disabled in lightweight mode)",
    }
    with open(vggt_dir / "wgeo_config.json", 'w') as f:
        json.dump(config, f, indent=2)
    print(f"[OK] W_geo 重算完成，配置保存到 {vggt_dir / 'wgeo_config.json'}")


if __name__ == "__main__":
    main()
