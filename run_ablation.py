"""
W_geo 系数消融实验自动化脚本
在 fern 场景上跑多组 (w_sigma, w_vis, w_cons) 配置
注：当前 fern 仅有 sigma+vis（无 _cons.npy），所有配置在同样降级模式下比较，公平。
"""
import os
import subprocess
import shutil
import json
from pathlib import Path

PY_3DGS = "D:/miniconda/envs/gaussian_splatting/python.exe"
ROOT = Path("d:/Downloads/研究")
SCENE = "llff_fern"
SCENE_DIR = ROOT / "vggt" / "examples" / SCENE
VGGT_DIR = SCENE_DIR / "vggt"
INPUT_DIR = SCENE_DIR / "3dgs_input"
OUT_BASE = ROOT / "output" / "ablation_wgeo"
OUT_BASE.mkdir(parents=True, exist_ok=True)

RECOMPUTE = ROOT / "vggt" / "recompute_wgeo.py"
CONVERT = ROOT / "vggt" / "convert_vggt_to_3dgs.py"
TRAIN = ROOT / "gaussian-splatting-main" / "train.py"

# 消融配置：(name, w_sigma, w_vis, w_cons)
# 第一类：基线（无权重图）+ 组件消融（看每个分量贡献）
# 第二类：参数敏感度（看默认值是否合理）
CONFIGS = [
    # 基线
    ("no_weight_baseline", 0.0, 0.0, 0.0),   # 无权重图（baseline）
    # 组件消融（看每个分量贡献）
    ("only_sigma",         1.0, 0.0, 0.0),   # 仅深度置信度
    ("sigma_plus_vis",     0.6, 0.4, 0.0),   # sigma + vis（去掉 C）
    ("vis_plus_cons",      0.0, 0.6, 0.4),   # vis + cons（去掉 σ）
    ("sigma_plus_cons",    0.6, 0.0, 0.4),   # sigma + cons（去掉 V）
    # 参数敏感度（三分量都开）
    ("default",            0.5, 0.3, 0.2),   # 默认（基线）
    ("equal_three",        0.33, 0.33, 0.33),# 均权
    ("sigma_dominant",     0.7, 0.2, 0.1),   # 偏深度
    ("vis_dominant",       0.2, 0.6, 0.2),   # 偏可见性
    ("cons_dominant",      0.2, 0.2, 0.6),   # 偏一致性
]

ITER = 7000
results = []

print(f"=== W_geo 系数消融实验 ({SCENE}, {ITER} iter) ===\n")

for name, ws, wv, wc in CONFIGS:
    print(f"\n{'='*60}")
    print(f"[{name}] w_sigma={ws}, w_vis={wv}, w_cons={wc}")
    print(f"{'='*60}")

    out_dir = OUT_BASE / f"{SCENE}_{name}"
    if out_dir.exists():
        shutil.rmtree(out_dir)

    # Step 1: 重算 W.npy（秒级）
    # baseline 模式跳过 W 重算（不会用到 weights/）
    if name != "no_weight_baseline":
        cmd_recomp = [
            PY_3DGS, str(RECOMPUTE),
            "--vggt_dir", str(VGGT_DIR),
            "--w_sigma", str(ws),
            "--w_vis", str(wv),
            "--w_cons", str(wc),
            "--w_strategy", "weighted_avg",
        ]
        print(f"\n[1/3] recompute W.npy ...")
        r = subprocess.run(cmd_recomp, capture_output=True, text=True, encoding='utf-8', errors='replace')
        if r.returncode != 0:
            err_msg = (r.stderr or "")[-500:]
            try:
                print(f"  ERROR: {err_msg.encode('ascii', errors='replace').decode('ascii')}")
            except Exception:
                print("  ERROR (encoding issue)")
            continue
    else:
        print(f"\n[1/3] skip W recompute (baseline)")

    # Step 2: 重跑 convert（生成 3dgs_input 目录里的 weights/ 是从 vggt/ 拷贝的）
    if INPUT_DIR.exists():
        shutil.rmtree(INPUT_DIR)
    cmd_conv = [
        PY_3DGS, str(CONVERT),
        "--vggt_dir", str(VGGT_DIR),
        "--output_dir", str(INPUT_DIR),
    ]
    print(f"[2/3] convert ...")
    r = subprocess.run(cmd_conv, capture_output=True, text=True, encoding='utf-8', errors='replace')
    if r.returncode != 0:
        err_msg = (r.stderr or "")[-500:]
        try:
            print(f"  ERROR convert: {err_msg.encode('ascii', errors='replace').decode('ascii')}")
        except Exception:
            print("  ERROR convert (encoding issue)")
        continue

    # Step 3: 训练 3DGS（4 分钟）
    cmd_train = [
        PY_3DGS, str(TRAIN),
        "-s", str(INPUT_DIR),
        "-d", "depths",
    ]
    # 当系数全为 0 时跳过权重图，等价于 baseline
    if name != "no_weight_baseline":
        cmd_train += ["--weights", "weights"]
    cmd_train += [
        "-m", str(out_dir),
        "--iterations", str(ITER),
        "--test_iterations", str(ITER),
        "--save_iterations", str(ITER),
        "--disable_viewer",
    ]
    print(f"[3/3] 训练 3DGS ({ITER} iter) ...")
    r = subprocess.run(cmd_train, capture_output=True, text=True, encoding='utf-8', errors='replace')

    # 解析输出获取 PSNR
    psnr = None
    l1 = None
    for line in (r.stdout + r.stderr).splitlines():
        if "Evaluating train" in line and f"ITER {ITER}" in line:
            try:
                # [ITER 7000] Evaluating train: L1 0.013 PSNR 31.05
                parts = line.split()
                idx_l1 = parts.index("L1")
                idx_psnr = parts.index("PSNR")
                l1 = float(parts[idx_l1 + 1])
                psnr = float(parts[idx_psnr + 1])
            except Exception as e:
                pass

    print(f"\n[Result] {name}: L1={l1}, PSNR={psnr}")
    results.append((name, ws, wv, wc, l1, psnr))

# 汇总
print(f"\n\n{'='*70}")
print(f"  W_geo 消融实验汇总（{SCENE}, {ITER} iter）")
print(f"{'='*70}")
print(f"{'Config':<22} {'(w_σ,w_V,w_C)':<22} {'L1 ↓':>10} {'PSNR ↑':>10}")
print("-" * 70)
for name, ws, wv, wc, l1, psnr in results:
    coef_str = f"({ws},{wv},{wc})"
    l1_str = f"{l1:.6f}" if l1 is not None else "N/A"
    psnr_str = f"{psnr:.4f}" if psnr is not None else "N/A"
    print(f"{name:<22} {coef_str:<22} {l1_str:>10} {psnr_str:>10}")

# 保存到 JSON
out_json = OUT_BASE / "ablation_summary.json"
with open(out_json, 'w', encoding='utf-8') as f:
    json.dump([{
        "config": name,
        "w_sigma": ws, "w_vis": wv, "w_cons": wc,
        "L1": l1, "PSNR": psnr
    } for name, ws, wv, wc, l1, psnr in results], f, indent=2, ensure_ascii=False)
print(f"\n[OK] 结果保存到: {out_json}")
