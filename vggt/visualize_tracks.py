#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
轨迹数据可视化脚本

用法:
    python visualize_tracks.py --tracks_dir vggt/examples/kitchen/vggt --images_dir vggt/examples/kitchen/images
    python visualize_tracks.py --tracks_dir vggt/examples/kitchen/vggt --images_dir vggt/examples/kitchen/images --frame_idx 0
    python visualize_tracks.py --tracks_dir vggt/examples/kitchen/vggt --images_dir vggt/examples/kitchen/images --show_connections
"""

import os
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm


def load_tracks_data(tracks_dir, frame_idx=None):
    """
    加载轨迹数据
    
    Args:
        tracks_dir: 轨迹数据目录
        frame_idx: 指定帧索引（None 表示加载所有帧）
    
    Returns:
        tracks_dict: {frame_idx: {"tracks": [N, 2], "vis_scores": [N], "confs": [N]}}
    """
    tracks_dict = {}
    tracks_dir = Path(tracks_dir)
    
    # 查找所有轨迹文件
    track_files = sorted(tracks_dir.glob("frame_*_tracks.npz"))
    
    if frame_idx is not None:
        # 只加载指定帧
        track_file = tracks_dir / f"frame_{frame_idx:03d}_tracks.npz"
        if track_file.exists():
            data = np.load(track_file)
            tracks_dict[frame_idx] = {
                "tracks": data["tracks"],
                "vis_scores": data.get("vis_scores", None),
                "confs": data.get("confs", None)
            }
    else:
        # 加载所有帧
        for track_file in track_files:
            # 从文件名提取帧索引
            frame_idx = int(track_file.stem.split("_")[1])
            data = np.load(track_file)
            tracks_dict[frame_idx] = {
                "tracks": data["tracks"],
                "vis_scores": data.get("vis_scores", None),
                "confs": data.get("confs", None)
            }
    
    return tracks_dict


def visualize_tracks_on_image(image, tracks, vis_scores=None, confs=None, 
                              color_by="vis", point_size=3, alpha=0.8):
    """
    在图像上绘制轨迹点
    
    Args:
        image: 图像数组 [H, W, 3] (BGR 格式)
        tracks: 轨迹点坐标 [N, 2] (x, y)
        vis_scores: 可见性分数 [N] (可选)
        confs: 置信度 [N] (可选)
        color_by: 颜色编码方式 ("vis", "conf", "fixed")
        point_size: 点的大小
        alpha: 透明度
    
    Returns:
        vis_image: 可视化后的图像
    """
    vis_image = image.copy()
    N = tracks.shape[0]
    
    if N == 0:
        return vis_image
    
    # 根据不同的编码方式设置颜色
    if color_by == "vis" and vis_scores is not None:
        # 使用可见性分数编码颜色（绿色=高可见性，红色=低可见性）
        colors = np.zeros((N, 3), dtype=np.uint8)
        for i in range(N):
            vis = vis_scores[i]
            # 绿色到红色的渐变
            colors[i] = [0, int(vis * 255), int((1 - vis) * 255)]
    elif color_by == "conf" and confs is not None:
        # 使用置信度编码颜色（蓝色=高置信度，红色=低置信度）
        colors = np.zeros((N, 3), dtype=np.uint8)
        conf_min, conf_max = confs.min(), confs.max()
        if conf_max > conf_min:
            conf_normalized = (confs - conf_min) / (conf_max - conf_min)
        else:
            conf_normalized = np.ones(N)
        for i in range(N):
            conf_norm = conf_normalized[i]
            # 蓝色到红色的渐变
            colors[i] = [int((1 - conf_norm) * 255), 0, int(conf_norm * 255)]
    else:
        # 固定颜色（黄色）
        colors = np.array([[0, 255, 255]] * N, dtype=np.uint8)  # 黄色 (BGR)
    
    # 绘制轨迹点
    for i in range(N):
        x, y = int(tracks[i, 0]), int(tracks[i, 1])
        if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
            color = tuple(map(int, colors[i]))
            cv2.circle(vis_image, (x, y), point_size, color, -1)
            # 添加半透明效果
            if alpha < 1.0:
                overlay = vis_image.copy()
                cv2.circle(overlay, (x, y), point_size, color, -1)
                cv2.addWeighted(overlay, 1 - alpha, vis_image, alpha, 0, vis_image)
    
    return vis_image


def visualize_tracks_connections(image, tracks_dict, frame_idx, 
                                 max_connections=100, line_thickness=1):
    """
    可视化跨帧的轨迹连接（如果有多帧数据）
    
    Args:
        image: 当前帧图像
        tracks_dict: 所有帧的轨迹数据
        frame_idx: 当前帧索引
        max_connections: 最多显示的连接数（避免过于密集）
        line_thickness: 线条粗细
    
    Returns:
        vis_image: 可视化后的图像
    """
    vis_image = image.copy()
    
    # 获取当前帧和相邻帧的轨迹
    if frame_idx not in tracks_dict:
        return vis_image
    
    current_tracks = tracks_dict[frame_idx]["tracks"]
    N = current_tracks.shape[0]
    
    # 只显示部分连接，避免过于密集
    if N > max_connections:
        indices = np.random.choice(N, max_connections, replace=False)
    else:
        indices = np.arange(N)
    
    # 绘制连接线（如果有下一帧）
    if frame_idx + 1 in tracks_dict:
        next_tracks = tracks_dict[frame_idx + 1]["tracks"]
        for i in indices:
            if i < next_tracks.shape[0]:
                pt1 = (int(current_tracks[i, 0]), int(current_tracks[i, 1]))
                pt2 = (int(next_tracks[i, 0]), int(next_tracks[i, 1]))
                if (0 <= pt1[0] < image.shape[1] and 0 <= pt1[1] < image.shape[0] and
                    0 <= pt2[0] < image.shape[1] and 0 <= pt2[1] < image.shape[0]):
                    cv2.line(vis_image, pt1, pt2, (0, 255, 0), line_thickness)
    
    return vis_image


def main():
    parser = argparse.ArgumentParser(description="可视化轨迹数据")
    parser.add_argument("--tracks_dir", type=str, required=True,
                       help="轨迹数据目录（包含 frame_XXX_tracks.npz 文件）")
    parser.add_argument("--images_dir", type=str, required=True,
                       help="图像目录")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="输出目录（默认：tracks_dir/track_visualizations）")
    parser.add_argument("--frame_idx", type=int, default=None,
                       help="指定帧索引（None 表示处理所有帧）")
    parser.add_argument("--color_by", type=str, default="vis",
                       choices=["vis", "conf", "fixed"],
                       help="颜色编码方式：vis（可见性）、conf（置信度）、fixed（固定颜色）")
    parser.add_argument("--point_size", type=int, default=3,
                       help="轨迹点的大小")
    parser.add_argument("--show_connections", action="store_true",
                       help="显示跨帧的轨迹连接")
    parser.add_argument("--max_points", type=int, default=None,
                       help="最多显示的轨迹点数（None 表示显示所有）")
    parser.add_argument("--vggt_resolution", type=int, default=518,
                       help="VGGT 处理的分辨率（默认 518）")
    parser.add_argument("--auto_scale", action="store_true", default=True,
                       help="自动缩放轨迹坐标到原始图像分辨率（默认启用）")
    
    args = parser.parse_args()
    
    # 设置输出目录
    if args.output_dir is None:
        args.output_dir = os.path.join(args.tracks_dir, "track_visualizations")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载轨迹数据
    print("加载轨迹数据...")
    tracks_dict = load_tracks_data(args.tracks_dir, args.frame_idx)
    
    if len(tracks_dict) == 0:
        print(f"未找到轨迹数据文件在 {args.tracks_dir}")
        return
    
    print(f"加载了 {len(tracks_dict)} 帧的轨迹数据")
    
    # 处理每一帧
    images_dir = Path(args.images_dir)
    frame_indices = sorted(tracks_dict.keys())
    
    for frame_idx in tqdm(frame_indices, desc="可视化轨迹"):
        # 加载图像
        image_files = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.JPG")) + \
                      sorted(images_dir.glob("*.png")) + sorted(images_dir.glob("*.PNG"))
        
        if frame_idx < len(image_files):
            image_path = image_files[frame_idx]
        else:
            print(f"警告: 帧 {frame_idx} 没有对应的图像文件")
            continue
        
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"警告: 无法加载图像: {image_path}")
            continue
        
        # 获取轨迹数据
        tracks_data = tracks_dict[frame_idx]
        tracks = tracks_data["tracks"]
        vis_scores = tracks_data.get("vis_scores", None)
        confs = tracks_data.get("confs", None)
        
        # 缩放轨迹坐标到原始图像分辨率
        if args.auto_scale:
            img_h, img_w = image.shape[:2]
            
            # 图像预处理流程：
            # 1. 原始图像 (W×H) -> 填充成正方形 (max(W,H)×max(W,H), center padding)
            # 2. 缩放到 1024×1024
            # 3. 缩放到 518×518 (VGGT 推理)
            # 4. 轨迹计算在 518×518 坐标系下
            
            # 计算填充偏移
            # 如果原始图像是 W×H，填充后是 max(W,H)×max(W,H)
            max_dim_original = max(img_w, img_h)
            
            # 在原始图像尺寸下的填充偏移
            if img_w >= img_h:
                # 横向图像，上下填充
                pad_left_original = 0
                pad_top_original = (max_dim_original - img_h) / 2
            else:
                # 纵向图像，左右填充
                pad_left_original = (max_dim_original - img_w) / 2
                pad_top_original = 0
            
            # 计算从填充后的正方形到 518×518 的缩放比例
            # 流程：max_dim×max_dim -> 1024×1024 -> 518×518
            # 总缩放比例 = 518 / max_dim
            scale_to_518 = args.vggt_resolution / max_dim_original
            
            # 在 518×518 坐标系下的填充偏移
            offset_x_518 = pad_left_original * scale_to_518
            offset_y_518 = pad_top_original * scale_to_518
            
            # 从 518×518 坐标映射回原始图像坐标
            # 注意：PyTorch interpolate 使用 align_corners=False（像素边缘对齐）
            # 而 PIL resize 使用像素中心对齐，需要调整
            
            # align_corners=False 到像素中心对齐的偏移
            # align_corners=False: 坐标范围 [0, W]，0 是左边缘，W 是右边缘
            # 像素中心对齐: 坐标范围 [0, W-1]，0 是第一个像素中心，W-1 是最后一个像素中心
            # 偏移量 = 0.5（半个像素）
            align_offset = 0.5
            
            # 缩放比例
            scale_x = img_w / args.vggt_resolution
            scale_y = img_h / args.vggt_resolution
            
            tracks = tracks.copy()
            # 坐标转换步骤：
            # 1. 从 align_corners=False 转换到像素中心对齐（减去 0.5）
            # 2. 减去填充偏移
            # 3. 缩放到原始图像尺寸
            # 4. 转换回像素中心对齐（加上 0.5）
            tracks[:, 0] = ((tracks[:, 0] - align_offset) - offset_x_518) * scale_x + align_offset
            tracks[:, 1] = ((tracks[:, 1] - align_offset) - offset_y_518) * scale_y + align_offset
            
            print(f"  帧 {frame_idx}: 缩放轨迹坐标 (518×518 -> {img_w}×{img_h}, 填充偏移: [{offset_x_518:.1f}, {offset_y_518:.1f}], align_offset: {align_offset})")
        
        # 限制显示的轨迹点数
        if args.max_points is not None and tracks.shape[0] > args.max_points:
            indices = np.random.choice(tracks.shape[0], args.max_points, replace=False)
            tracks = tracks[indices]
            if vis_scores is not None:
                vis_scores = vis_scores[indices]
            if confs is not None:
                confs = confs[indices]
        
        # 可视化轨迹点
        vis_image = visualize_tracks_on_image(
            image, tracks, vis_scores, confs,
            color_by=args.color_by,
            point_size=args.point_size
        )
        
        # 显示连接（如果启用）
        if args.show_connections:
            vis_image = visualize_tracks_connections(
                vis_image, tracks_dict, frame_idx
            )
        
        # 保存结果
        output_path = os.path.join(args.output_dir, f"frame_{frame_idx:03d}_tracks_vis.png")
        cv2.imwrite(output_path, vis_image)
        
        # 创建图例（仅第一帧）
        if frame_idx == frame_indices[0]:
            create_legend(args.output_dir, args.color_by, vis_scores, confs)
    
    print(f"\n可视化完成！结果保存在: {args.output_dir}")


def create_legend(output_dir, color_by, vis_scores=None, confs=None):
    """创建颜色图例"""
    fig, ax = plt.subplots(figsize=(6, 1))
    ax.axis('off')
    
    if color_by == "vis" and vis_scores is not None:
        # 可见性图例
        gradient = np.linspace(0, 1, 100).reshape(1, -1)
        ax.imshow(gradient, aspect='auto', cmap='RdYlGn', extent=[0, 1, 0, 1])
        ax.text(0.5, -0.3, '可见性分数 (0=低, 1=高)', ha='center', transform=ax.transAxes)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    elif color_by == "conf" and confs is not None:
        # 置信度图例
        gradient = np.linspace(0, 1, 100).reshape(1, -1)
        ax.imshow(gradient, aspect='auto', cmap='coolwarm', extent=[0, 1, 0, 1])
        ax.text(0.5, -0.3, '置信度 (0=低, 1=高)', ha='center', transform=ax.transAxes)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    else:
        ax.text(0.5, 0.5, '固定颜色（黄色）', ha='center', va='center', transform=ax.transAxes)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "legend.png"), dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    main()

