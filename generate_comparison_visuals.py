"""
生成 baseline vs W_geo vs GT 视觉对比图
为每个场景生成：
  1. 三联横向拼图：GT | Baseline | W_geo
  2. 误差热力图：|render - gt| 的可视化（红色越深 = 误差越大）
"""
import os
from pathlib import Path
import cv2
import numpy as np

OUTPUT_DIR = Path("d:/Downloads/研究/output")
VIS_DIR = Path("d:/Downloads/研究/comparison_visuals")
VIS_DIR.mkdir(exist_ok=True)

# 每个场景的 baseline 和 W_geo 输出目录
SCENES = {
    "kitchen": ("厨房_无权重_基线_38.66dB", "厨房_W几何_深度加权_39.23dB"),
    "fern": ("蕨类_无权重_基线_27.81dB", "蕨类_W几何_深度加权_29.44dB"),
    "flower": ("花卉_无权重_基线_21.41dB", "花卉_W几何_深度加权_21.52dB"),
}

# 选每个场景哪一帧做对比（可以全画或选有代表性的）
FRAMES_TO_SHOW = ["00000.png", "00001.png", "00002.png", "00003.png", "00004.png"]

def imread_unicode(path):
    """支持 Unicode 路径的 imread"""
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)

def imwrite_unicode(path, img):
    """支持 Unicode 路径的 imwrite"""
    success, buf = cv2.imencode(os.path.splitext(str(path))[1], img)
    if success:
        buf.tofile(str(path))

def make_label(text, width, height=40, color=(255, 255, 255), bg=(50, 50, 50)):
    """生成标签条"""
    label = np.full((height, width, 3), bg, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, 0.7, 2)[0]
    x = (width - text_size[0]) // 2
    y = (height + text_size[1]) // 2
    cv2.putText(label, text, (x, y), font, 0.7, color, 2, cv2.LINE_AA)
    return label

def compute_error_heatmap(render, gt):
    """计算误差热力图（红色越深 = 误差越大）"""
    diff = np.abs(render.astype(np.float32) - gt.astype(np.float32))
    err = diff.mean(axis=2)
    # 归一化到 0-1（用 95 百分位避免离群值）
    p95 = np.percentile(err, 95) if err.max() > 0 else 1.0
    err_norm = np.clip(err / max(p95, 1e-6), 0, 1)
    # 用 jet colormap：蓝色（误差小）→ 红色（误差大）
    err_uint8 = (err_norm * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(err_uint8, cv2.COLORMAP_JET)
    return heatmap, err.mean()

def make_comparison(scene_key, baseline_dir, wgeo_dir, frame_name):
    """为一个场景的一帧生成对比图"""
    base_render_dir = OUTPUT_DIR / baseline_dir / "train" / "ours_7000"
    wgeo_render_dir = OUTPUT_DIR / wgeo_dir / "train" / "ours_7000"

    gt = imread_unicode(base_render_dir / "gt" / frame_name)
    base = imread_unicode(base_render_dir / "renders" / frame_name)
    wgeo = imread_unicode(wgeo_render_dir / "renders" / frame_name)

    if gt is None or base is None or wgeo is None:
        print(f"[skip] {scene_key}/{frame_name}: missing image")
        return None

    H, W = gt.shape[:2]

    # 计算误差热力图
    base_heat, base_err = compute_error_heatmap(base, gt)
    wgeo_heat, wgeo_err = compute_error_heatmap(wgeo, gt)

    # 上排：GT | Baseline | W_geo（渲染图）
    # 下排：（空）| Baseline 误差图 | W_geo 误差图
    label_h = 40
    gt_label = make_label("Ground Truth", W, label_h)
    base_label = make_label(f"Baseline (L1={base_err:.1f})", W, label_h)
    wgeo_label = make_label(f"W_geo (L1={wgeo_err:.1f})", W, label_h, color=(0, 255, 255))

    top_row = np.concatenate([
        np.concatenate([gt_label, gt], axis=0),
        np.concatenate([base_label, base], axis=0),
        np.concatenate([wgeo_label, wgeo], axis=0),
    ], axis=1)

    # 左下：色阶图例（colorbar） - 解释红蓝代表什么
    legend = make_color_legend(W, H)
    legend_label = make_label("Error Scale (Blue=Low, Red=High)", W, label_h, bg=(80, 30, 30))

    bottom_row = np.concatenate([
        np.concatenate([legend_label, legend], axis=0),
        np.concatenate([make_label("Baseline Error", W, label_h, bg=(80, 30, 30)), base_heat], axis=0),
        np.concatenate([make_label("W_geo Error", W, label_h, bg=(80, 30, 30)), wgeo_heat], axis=0),
    ], axis=1)

    full = np.concatenate([top_row, bottom_row], axis=0)
    return full

def make_color_legend(width, height):
    """生成 jet colormap 色阶图例"""
    # 创建一个垂直的渐变色条
    bar_width = width // 4
    bar_height = int(height * 0.7)
    bar_x = (width - bar_width) // 2
    bar_y = (height - bar_height) // 2

    # 渐变值：从 0（顶部，蓝）到 255（底部，红）—— 注意要倒过来才符合"红=误差大"
    gradient = np.linspace(0, 255, bar_height).astype(np.uint8)
    gradient = np.tile(gradient.reshape(-1, 1), (1, bar_width))
    gradient_color = cv2.applyColorMap(gradient, cv2.COLORMAP_JET)

    # 整体背景
    canvas = np.full((height, width, 3), 30, dtype=np.uint8)
    canvas[bar_y:bar_y+bar_height, bar_x:bar_x+bar_width] = gradient_color

    # 加文字标注
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, "Low Error", (bar_x + bar_width + 10, bar_y + 20),
                font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, "(Better)", (bar_x + bar_width + 10, bar_y + 40),
                font, 0.4, (200, 255, 200), 1, cv2.LINE_AA)
    cv2.putText(canvas, "High Error", (bar_x + bar_width + 10, bar_y + bar_height - 10),
                font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, "(Worse)", (bar_x + bar_width + 10, bar_y + bar_height + 10),
                font, 0.4, (200, 200, 255), 1, cv2.LINE_AA)
    return canvas

def main():
    for scene_key, (baseline_dir, wgeo_dir) in SCENES.items():
        scene_out = VIS_DIR / scene_key
        scene_out.mkdir(exist_ok=True)
        print(f"\n=== {scene_key} ===")

        for frame_name in FRAMES_TO_SHOW:
            comp = make_comparison(scene_key, baseline_dir, wgeo_dir, frame_name)
            if comp is None:
                continue
            out_path = scene_out / f"compare_{frame_name}"
            imwrite_unicode(out_path, comp)
            print(f"  saved: {out_path}")

    print(f"\nDone. Output: {VIS_DIR}")

if __name__ == "__main__":
    main()
