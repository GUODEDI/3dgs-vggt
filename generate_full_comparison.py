"""
生成 5 方案完整对比图：
GT | Baseline | W_geo | W_geo+W_appear (precomp) | W_geo+W_appear (online) | Auto
"""
import os
from pathlib import Path
import cv2
import numpy as np

OUTPUT_DIR = Path("d:/Downloads/研究/output")
VIS_DIR = Path("d:/Downloads/研究/comparison_visuals_full")
VIS_DIR.mkdir(exist_ok=True)

# 每个场景：(显示名, 输出文件夹名)
SCENES = {
    "kitchen": [
        ("Baseline",         "厨房_无权重_基线_38.66dB",          40.34),
        ("W_geo",            "厨房_W几何_深度加权_39.23dB",       40.40),
        ("W_geo+W_app(pre)", "厨房_W几何+W外观_预计算_33.75dB",   None),
        ("W_geo+W_app(on)",  "厨房_W几何+W外观_在线_35.00dB",     None),
        ("Auto",             "厨房_自动模式_最终_39.17dB",        None),
    ],
    "fern": [
        ("Baseline",         "蕨类_无权重_基线_27.81dB",          29.13),
        ("W_geo",            "蕨类_W几何_深度加权_29.44dB",       31.05),
        ("W_geo+W_app(pre)", "蕨类_W几何+W外观_预计算_26.53dB",   None),
        ("W_geo+W_app(on)",  "蕨类_W几何+W外观_在线_27.71dB",     None),
        ("Auto",             "蕨类_自动模式_最终_29.41dB",        None),
    ],
    "flower": [
        ("Baseline",         "花卉_无权重_基线_21.41dB",          21.25),
        ("W_geo",            "花卉_W几何_深度加权_21.52dB",       21.75),
        ("W_geo+W_app(pre)", "花卉_W几何+W外观_预计算_21.96dB",   None),
        ("W_geo+W_app(on)",  "花卉_W几何+W外观_在线_21.30dB",     None),
        ("Auto",             "花卉_自动模式_最终_21.46dB",        None),
    ],
}

FRAMES = ["00000.png", "00001.png", "00002.png", "00003.png", "00004.png"]

def imread_unicode(path):
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)

def imwrite_unicode(path, img):
    success, buf = cv2.imencode(os.path.splitext(str(path))[1], img)
    if success:
        buf.tofile(str(path))

def make_label(text, width, height=40, color=(255, 255, 255), bg=(50, 50, 50)):
    label = np.full((height, width, 3), bg, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, 0.6, 2)[0]
    x = max((width - text_size[0]) // 2, 5)
    y = (height + text_size[1]) // 2
    cv2.putText(label, text, (x, y), font, 0.6, color, 2, cv2.LINE_AA)
    return label

def compute_error_heatmap(render, gt):
    diff = np.abs(render.astype(np.float32) - gt.astype(np.float32))
    err = diff.mean(axis=2)
    p95 = np.percentile(err, 95) if err.max() > 0 else 1.0
    err_norm = np.clip(err / max(p95, 1e-6), 0, 1)
    err_uint8 = (err_norm * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(err_uint8, cv2.COLORMAP_JET)
    return heatmap, err.mean()

def make_comparison_full(scene_key, methods, frame_name):
    """5 方案横向对比"""
    # 读 GT（用 baseline 目录的 gt）
    gt_path = OUTPUT_DIR / methods[0][1] / "train" / "ours_7000" / "gt" / frame_name
    gt = imread_unicode(gt_path)
    if gt is None:
        return None
    H, W = gt.shape[:2]
    label_h = 35

    # 收集所有渲染图
    renders = []
    errors = []
    for name, dir_name, _ in methods:
        r_path = OUTPUT_DIR / dir_name / "train" / "ours_7000" / "renders" / frame_name
        r = imread_unicode(r_path)
        if r is None:
            renders.append(np.full((H, W, 3), 50, dtype=np.uint8))
            errors.append(None)
            continue
        renders.append(r)
        heat, err = compute_error_heatmap(r, gt)
        errors.append((heat, err))

    # 顶行：GT + 5 个渲染图
    gt_label = make_label("Ground Truth", W, label_h, color=(0, 255, 0))
    top_imgs = [np.concatenate([gt_label, gt], axis=0)]
    for (name, _, _), r, err in zip(methods, renders, errors):
        err_str = f" L1={err[1]:.1f}" if err else ""
        lbl = make_label(name + err_str, W, label_h)
        top_imgs.append(np.concatenate([lbl, r], axis=0))
    top_row = np.concatenate(top_imgs, axis=1)

    # 底行：色阶图例 + 5 个误差热力图
    legend = make_color_legend(W, H)
    legend_label = make_label("Error: Blue=Low, Red=High", W, label_h, bg=(80, 30, 30))
    bot_imgs = [np.concatenate([legend_label, legend], axis=0)]
    for (name, _, _), err in zip(methods, errors):
        lbl = make_label(name + " Error", W, label_h, bg=(80, 30, 30))
        if err:
            bot_imgs.append(np.concatenate([lbl, err[0]], axis=0))
        else:
            bot_imgs.append(np.concatenate([lbl, np.full((H, W, 3), 50, dtype=np.uint8)], axis=0))
    bot_row = np.concatenate(bot_imgs, axis=1)

    return np.concatenate([top_row, bot_row], axis=0)

def make_color_legend(width, height):
    bar_width = width // 4
    bar_height = int(height * 0.7)
    bar_x = (width - bar_width) // 2
    bar_y = (height - bar_height) // 2
    gradient = np.linspace(0, 255, bar_height).astype(np.uint8)
    gradient = np.tile(gradient.reshape(-1, 1), (1, bar_width))
    gradient_color = cv2.applyColorMap(gradient, cv2.COLORMAP_JET)
    canvas = np.full((height, width, 3), 30, dtype=np.uint8)
    canvas[bar_y:bar_y+bar_height, bar_x:bar_x+bar_width] = gradient_color
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, "Low", (bar_x + bar_width + 5, bar_y + 20),
                font, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, "High", (bar_x + bar_width + 5, bar_y + bar_height - 5),
                font, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas

def main():
    for scene_key, methods in SCENES.items():
        scene_out = VIS_DIR / scene_key
        scene_out.mkdir(exist_ok=True)
        print(f"\n=== {scene_key} ===")
        for frame_name in FRAMES:
            comp = make_comparison_full(scene_key, methods, frame_name)
            if comp is None:
                continue
            out_path = scene_out / f"compare_{frame_name}"
            imwrite_unicode(out_path, comp)
            print(f"  saved: {out_path.name}")
    print(f"\nDone. Output: {VIS_DIR}")

if __name__ == "__main__":
    main()
