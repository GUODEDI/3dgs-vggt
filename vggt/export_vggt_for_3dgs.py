#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
VGGT → 3DGS 数据导出工具

对每个场景运行一次 VGGT，产出后续 3DGS 训练所需的所有"护航信息"并落盘。
"""

import os
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2

# 注意：expandable_segments 在 Windows 上不支持，已移除
from PIL import Image

# VGGT imports
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images_square
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.dependency.projection import project_3D_points_np


def parse_args():
    parser = argparse.ArgumentParser(description="VGGT → 3DGS 数据导出工具（GPU）")
    parser.add_argument("--scene_dir", type=str, required=True, help="场景目录（包含 images/ 子目录）")
    parser.add_argument("--output_dir", type=str, default=None, help="输出目录（默认：scene_dir）")
    parser.add_argument("--num_frames", type=int, default=8, help="处理的帧数（默认 8，受 GPU 显存限制）")

    # W_geo 权重图融合参数（修士论文需要做消融实验）
    parser.add_argument("--w_sigma", type=float, default=0.5, help="W_geo 中深度可靠性分量权重（默认 0.5）")
    parser.add_argument("--w_vis", type=float, default=0.3, help="W_geo 中可见性分量权重（默认 0.3）")
    parser.add_argument("--w_cons", type=float, default=0.2, help="W_geo 中一致性分量权重（默认 0.2）")
    parser.add_argument("--w_strategy", type=str, default="weighted_avg",
                        choices=["multiplicative", "weighted_avg"],
                        help="W_geo 融合策略（默认 weighted_avg）")
    parser.add_argument("--w_min", type=float, default=0.1, help="W_geo 最小裁剪阈值（默认 0.1）")
    return parser.parse_args()


def setup_device():
    """设置计算设备（固定使用 CUDA）"""
    if not torch.cuda.is_available():
        raise ValueError("CUDA 不可用，此脚本需要 GPU")
    
    device = "cuda"
    print(f"✅ 使用 CUDA GPU 加速")
    print(f"   GPU 名称: {torch.cuda.get_device_name(0)}")
    print(f"   CUDA 版本: {torch.version.cuda}")
    print(f"   GPU 内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    return device


def load_model(device):
    """加载 VGGT 模型"""
    print("加载 VGGT 模型...")
    
    # 清理 GPU 缓存（在加载模型前）
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print("✓ GPU 缓存已清理")
    
    model = VGGT()
    
    # 尝试从 vggt/models/ 目录加载
    script_dir = Path(__file__).parent
    local_model_path = script_dir / "models" / "model.pt"
    
    if local_model_path.exists():
        print(f"从本地目录加载模型: {local_model_path}")
        model.load_state_dict(torch.load(local_model_path, map_location='cpu', mmap=True))
    else:
        # 自动下载到 vggt/models/ 目录
        _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
        print(f"下载模型到: {local_model_path.parent}")
        local_model_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"正在下载模型（约 4.68 GB）...")
        torch.hub.download_url_to_file(_URL, str(local_model_path))
        print(f"模型下载完成: {local_model_path}")
        model.load_state_dict(torch.load(local_model_path, map_location='cpu', mmap=True))
    
    model.eval()
    
    # 将模型移到 GPU（使用更稳健的方式）
    if device == "cuda":
        print("将模型移到 GPU...")
        # 多次清理以确保内存释放
        for _ in range(3):
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        
        # 添加短暂延迟让系统释放内存
        import time
        time.sleep(0.5)
        
        try:
            # 使用 with 语句确保在正确的设备上下文中
            with torch.cuda.device(device):
                model = model.to(device)
        except torch.cuda.OutOfMemoryError as e:
            print(f"⚠️  GPU 内存不足: {e}")
            print("⚠️  尝试清理缓存后重试...")
            # 更激进的清理
            for _ in range(5):
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            time.sleep(1.0)  # 等待更长时间
            
            # 再次尝试
            try:
                with torch.cuda.device(device):
                    model = model.to(device)
            except torch.cuda.OutOfMemoryError:
                print("⚠️  仍然内存不足，建议：")
                print("   1. 关闭其他占用 GPU 的程序（特别是 Chrome、Cursor、微信等）")
                print("   2. 重启 Python 环境以完全释放 GPU 内存")
                print("   3. 运行 'nvidia-smi' 检查是否有其他 Python 进程占用 GPU")
                print("   4. 如果问题持续，可能需要重启计算机以完全清理 GPU 内存碎片")
                raise RuntimeError("GPU 内存不足，无法加载模型。请关闭其他占用 GPU 的程序。")
    
    print("模型加载完成")
    return model


def run_vggt_inference(model, images, device, resolution=518, batch_size=-1):
    """
    运行 VGGT 推理（一次性处理所有图像）
    
    Args:
        model: VGGT 模型
        images: [S, 3, H, W] 图像张量
        device: 计算设备
        resolution: VGGT 内部分辨率（默认 518）
        batch_size: 已废弃，保留参数以兼容
    
    Returns:
        extrinsic: [S, 3, 4] 外参矩阵
        intrinsic: [S, 3, 3] 内参矩阵
        depth_map: [S, H, W, 1] 深度图
        depth_conf: [S, H, W] 深度置信度
    """
    print(f"运行 VGGT 推理（一次性处理 {images.shape[0]} 帧）...")
    return _run_vggt_inference_single(model, images, device, resolution)


def _run_vggt_inference_single(model, images, device, resolution=518):
    """单批 VGGT 推理（内部函数）"""
    with tqdm(total=3, desc="VGGT 推理", unit="步骤", leave=False) as pbar:
        images_resized = F.interpolate(images, size=(resolution, resolution), mode="bilinear", align_corners=False)
        pbar.update(1)
        pbar.set_description("VGGT 推理: 特征提取")
        
        # CUDA 使用 autocast
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        use_autocast = True
        
        with torch.no_grad():
            if use_autocast:
                autocast_context = torch.cuda.amp.autocast(dtype=dtype, enabled=True)
                with autocast_context:
                    images_batch = images_resized[None]
                    aggregated_tokens_list, ps_idx = model.aggregator(images_batch)
            else:
                images_batch = images_resized[None]
                aggregated_tokens_list, ps_idx = model.aggregator(images_batch)
            
            pbar.update(1)
            pbar.set_description("VGGT 推理: 预测相机和深度")
            
            if use_autocast:
                with autocast_context:
                    pose_enc = model.camera_head(aggregated_tokens_list)[-1]
                    extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images_batch.shape[-2:])
                    depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images_batch, ps_idx)
            else:
                pose_enc = model.camera_head(aggregated_tokens_list)[-1]
                extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images_batch.shape[-2:])
                depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images_batch, ps_idx)
            
            pbar.update(1)
            pbar.set_description("VGGT 推理: 完成")
    
    # 转换为 numpy
    extrinsic = extrinsic.squeeze(0).cpu().numpy()
    intrinsic = intrinsic.squeeze(0).cpu().numpy()
    depth_map = depth_map.squeeze(0).cpu().numpy()
    depth_conf = depth_conf.squeeze(0).cpu().numpy()
    
    return extrinsic, intrinsic, depth_map, depth_conf


def compute_uncertainty(depth_conf, percentile_low=5, percentile_high=95):
    """
    从深度置信度计算不确定度
    
    Args:
        depth_conf: [H, W] 深度置信度（值越大越可信）
        percentile_low: 低分位数（默认 5%）
        percentile_high: 高分位数（默认 95%）
    
    Returns:
        sigma_normalized: [H, W] 归一化的不确定度（值越大越不可信，范围 0-1）
    """
    # depth_conf 值越大越可信，转换为不确定度（值越大越不可信）
    # 使用倒数映射
    eps = 1e-8
    sigma = 1.0 / (depth_conf + eps)
    
    # 分位数归一化
    p_low = np.percentile(sigma, percentile_low)
    p_high = np.percentile(sigma, percentile_high)
    
    if p_high > p_low:
        sigma_normalized = (sigma - p_low) / (p_high - p_low)
        sigma_normalized = np.clip(sigma_normalized, 0.0, 1.0)
    else:
        sigma_normalized = np.zeros_like(sigma)
    
    return sigma_normalized


def compute_visibility_from_tracks(pred_vis_scores, pred_tracks, image_shape):
    """
    从稀疏轨迹可见性插值到全图可见性
    
    Args:
        pred_vis_scores: [S, N] 稀疏可见性分数
        pred_tracks: [S, N, 2] 轨迹点坐标
        image_shape: (H, W) 图像尺寸
    
    Returns:
        vis_map: [S, H, W] 全图可见性图
    """
    S, H, W = pred_vis_scores.shape[0], image_shape[0], image_shape[1]
    vis_map = np.ones((S, H, W), dtype=np.float32)
    
    # 创建像素坐标网格
    y_coords, x_coords = np.mgrid[0:H, 0:W]
    grid_points = np.stack([x_coords.flatten(), y_coords.flatten()], axis=1).astype(np.float32)
    
    for s in tqdm(range(S), desc="插值可见性", unit="帧", leave=False):
        # 获取当前帧的可见性分数和轨迹点
        vis_scores = pred_vis_scores[s]  # [N]
        track_points = pred_tracks[s]  # [N, 2]
        
        if len(vis_scores) == 0:
            continue
        
        # 使用分批处理避免内存溢出
        # 对于每个像素，只考虑附近的轨迹点（k-NN 方法）
        batch_size = 10000  # 每次处理 10000 个像素
        vis_interp = np.zeros(H * W, dtype=np.float32)
        
        # 使用 float32 减少内存使用
        grid_points_f32 = grid_points.astype(np.float32)
        track_points_f32 = track_points.astype(np.float32)
        vis_scores_f32 = vis_scores.astype(np.float32)
        
        # 只考虑最近的 k 个点，而不是所有点
        k = min(50, len(track_points))  # 最多考虑 50 个最近的点
        sigma = 10.0  # 控制插值范围
        
        for i in range(0, H * W, batch_size):
            end_idx = min(i + batch_size, H * W)
            batch_grid = grid_points_f32[i:end_idx]  # [batch_size, 2]
            
            # 计算到所有轨迹点的距离（使用 float32）
            distances_batch = np.sqrt(
                ((batch_grid[:, None, :] - track_points_f32[None, :, :]) ** 2).sum(axis=2)
            ).astype(np.float32)  # [batch_size, N]
            
            # 只使用最近的 k 个点进行插值
            if k < len(track_points):
                # 找到最近的 k 个点
                nearest_indices = np.argpartition(distances_batch, k, axis=1)[:, :k]  # [batch_size, k]
                # 获取对应的距离和可见性分数
                batch_distances = np.take_along_axis(distances_batch, nearest_indices, axis=1)
                batch_vis = np.take_along_axis(vis_scores_f32[None, :], nearest_indices, axis=1)
            else:
                batch_distances = distances_batch
                batch_vis = vis_scores_f32[None, :]
            
            # 使用高斯权重进行距离加权平均
            weights = np.exp(-batch_distances**2 / (2 * sigma**2)).astype(np.float32)
            weights = weights / (weights.sum(axis=1, keepdims=True) + 1e-8)  # 归一化
            
            vis_interp[i:end_idx] = (weights * batch_vis).sum(axis=1)
        
        vis_map[s] = vis_interp.reshape(H, W)
    
    return vis_map


def compute_consistency_from_tracks(world_points, pred_tracks, extrinsics, intrinsics):
    """
    从轨迹计算重投影误差，转换为一致性分数
    
    Args:
        world_points: [S, H, W, 3] 3D 世界坐标点
        pred_tracks: [S, N, 2] 轨迹点
        extrinsics: [S, 3, 4] 外参矩阵
        intrinsics: [S, 3, 3] 内参矩阵
    
    Returns:
        consistency: [S, N] 一致性分数（误差小 → 分数高）
    """
    S, N = pred_tracks.shape[0], pred_tracks.shape[1]
    consistency = np.ones((S, N), dtype=np.float32)
    
    # 从第一帧的轨迹点获取 3D 坐标
    if S < 2:
        return consistency
    
    # 简化：使用第一帧作为参考
    ref_frame = 0
    ref_tracks = pred_tracks[ref_frame]  # [N, 2]
    
    # 从参考帧的深度图获取 3D 点
    # 注意：使用双线性插值保持亚像素精度，而不是直接索引
    H, W = world_points.shape[1:3]
    ref_world_points = world_points[ref_frame]  # [H, W, 3]
    
    # 使用双线性插值获取轨迹点位置的 3D 点
    try:
        from scipy.interpolate import griddata
        y_grid, x_grid = np.mgrid[0:H, 0:W]
        points_grid = np.column_stack([x_grid.ravel(), y_grid.ravel()])  # [H*W, 2]
        world_points_flat = ref_world_points.reshape(-1, 3)  # [H*W, 3]
        
        ref_3d_points = griddata(
            points_grid, world_points_flat,
            ref_tracks, method='linear', fill_value=np.nan
        )  # [N, 3]
        
        # 对于插值失败的点，使用最近邻插值作为后备
        valid_mask = ~np.isnan(ref_3d_points).any(axis=1)
        if valid_mask.sum() < N * 0.9:
            ref_3d_points_fallback = griddata(
                points_grid, world_points_flat,
                ref_tracks[~valid_mask], method='nearest', fill_value=0
            )
            ref_3d_points[~valid_mask] = ref_3d_points_fallback
            valid_mask = ~np.isnan(ref_3d_points).any(axis=1)
        
        # 只使用有效的点进行后续计算
        ref_3d_points_valid = ref_3d_points[valid_mask]
        ref_tracks_valid = ref_tracks[valid_mask]
    except ImportError:
        # 如果没有 scipy，使用直接索引（精度较低）
        ref_tracks_int = ref_tracks.astype(int)
        ref_tracks_int[:, 0] = np.clip(ref_tracks_int[:, 0], 0, W-1)
        ref_tracks_int[:, 1] = np.clip(ref_tracks_int[:, 1], 0, H-1)
        ref_3d_points = world_points[ref_frame, ref_tracks_int[:, 1], ref_tracks_int[:, 0]]  # [N, 3]
        ref_3d_points_valid = ref_3d_points
        ref_tracks_valid = ref_tracks
        valid_mask = np.ones(N, dtype=bool)
    
    # 计算重投影误差（只使用有效的点）
    for s in range(1, S):
        # 将 3D 点投影到当前帧
        # project_3D_points_np 期望 points3D 是 (N, 3)，extrinsics 是 (B, 3, 4)
        points_2d, _ = project_3D_points_np(
            ref_3d_points_valid,  # [N_valid, 3] - 只使用有效的点
            extrinsics[s:s+1],     # [1, 3, 4]
            intrinsics[s:s+1]       # [1, 3, 3]
        )
        points_2d = points_2d[0]  # [N_valid, 2] - 从 (1, N_valid, 2) 中取出第一帧
        
        # 计算与轨迹的误差（只使用有效的点）
        reproj_error_valid = np.linalg.norm(points_2d - pred_tracks[s][valid_mask], axis=1)  # [N_valid]
        
        # 创建完整的误差数组（无效点设为大误差）
        reproj_error = np.full(N, np.nan)
        reproj_error[valid_mask] = reproj_error_valid
        
        # 误差小 → 一致性高
        eps = 1e-8
        consistency[s] = 1.0 / (reproj_error + eps)
    
    # 归一化
    for s in range(S):
        if consistency[s].max() > consistency[s].min():
            consistency[s] = (consistency[s] - consistency[s].min()) / (consistency[s].max() - consistency[s].min() + 1e-8)
    
    return consistency


def compute_weight_map(sigma_reliability, vis_reliability, consistency_reliability, 
                       w_sigma, w_vis, w_cons, strategy="multiplicative", min_weight=0.1):
    """
    计算可信像素权重图
    
    Args:
        sigma_reliability: [H, W] 不确定度可靠性（越大越可信）
        vis_reliability: [H, W] 可见性可靠性（越大越可信）
        consistency_reliability: [H, W] 一致性可靠性（越大越可信，可选）
        w_sigma: 不确定度权重
        w_vis: 可见性权重
        w_cons: 一致性权重
        strategy: 融合策略（multiplicative 或 weighted_avg）
        min_weight: 最小权重阈值
    
    Returns:
        W: [H, W] 权重图（0-1）
    """
    if consistency_reliability is None:
        # 只使用不确定度和可见性（轨迹缺失的降级情况）
        if strategy == "multiplicative":
            W = (sigma_reliability ** w_sigma) * (vis_reliability ** w_vis)
        else:  # weighted_avg
            total_weight = w_sigma + w_vis
            W = (w_sigma / total_weight) * sigma_reliability + (w_vis / total_weight) * vis_reliability
    else:
        if strategy == "multiplicative":
            W = (sigma_reliability ** w_sigma) * (vis_reliability ** w_vis) * (consistency_reliability ** w_cons)
        else:  # weighted_avg
            # 修复：归一化系数，避免用户传入非和为 1 的系数时输出 > 1
            total_weight = w_sigma + w_vis + w_cons
            W = ((w_sigma / total_weight) * sigma_reliability
                 + (w_vis / total_weight) * vis_reliability
                 + (w_cons / total_weight) * consistency_reliability)

    # 最小阈值裁剪
    W = np.maximum(W, min_weight)
    W = np.clip(W, 0.0, 1.0)
    
    return W


def percentile_normalize(data, p_low=5, p_high=95):
    """分位数归一化到 0-1"""
    p_low_val = np.percentile(data, p_low)
    p_high_val = np.percentile(data, p_high)
    
    if p_high_val > p_low_val:
        normalized = (data - p_low_val) / (p_high_val - p_low_val)
        normalized = np.clip(normalized, 0.0, 1.0)
    else:
        # 如果所有值相同，返回全 1（表示都可信）而不是全 0
        normalized = np.ones_like(data)
    
    return normalized


def save_camera_params(frame_idx, extrinsic, intrinsic, image_size, output_path, original_image_path):
    """保存相机参数到 JSON"""
    # 提取内参
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    
    # 提取外参
    R = extrinsic[:3, :3].tolist()
    t = extrinsic[:3, 3].tolist()
    
    camera_data = {
        "intrinsic": {
            "fx": float(fx),
            "fy": float(fy),
            "cx": float(cx),
            "cy": float(cy),
            "matrix": intrinsic.tolist()
        },
        "extrinsic": {
            "R": R,
            "t": t,
            "matrix": extrinsic.tolist()
        },
        "image_size": [int(image_size[0]), int(image_size[1])],
        "original_image_path": original_image_path,
        "coordinate_system": "OpenCV"  # x-right, y-down, z-forward
    }
    
    with open(output_path, 'w') as f:
        json.dump(camera_data, f, indent=2)


def create_visualization(depth, sigma, W, output_dir, frame_name):
    """创建可视化预览图"""
    # 深度伪彩图
    depth_normalized = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    depth_colored = cm.viridis(depth_normalized)[:, :, :3]
    depth_colored = (depth_colored * 255).astype(np.uint8)
    plt.imsave(os.path.join(output_dir, f"{frame_name}_preview_depth.png"), depth_colored)
    
    # 不确定度热力图
    sigma_colored = cm.hot(sigma)[:, :, :3]
    sigma_colored = (sigma_colored * 255).astype(np.uint8)
    plt.imsave(os.path.join(output_dir, f"{frame_name}_preview_sigma.png"), sigma_colored)
    
    # 权重图热力图
    W_colored = cm.viridis(W)[:, :, :3]
    W_colored = (W_colored * 255).astype(np.uint8)
    plt.imsave(os.path.join(output_dir, f"{frame_name}_preview_W.png"), W_colored)


def visualize_tracks_on_image(image_path, tracks, vis_scores=None, confs=None, 
                               original_coords=None, vggt_resolution=518,
                               point_size=3, max_points=2000):
    """
    在原始图像上绘制轨迹点
    
    Args:
        image_path: 原始图像路径
        tracks: 轨迹点坐标 [N, 2] (在 VGGT 分辨率下的坐标)
        vis_scores: 可见性分数 [N] (可选)
        confs: 置信度 [N] (可选)
        original_coords: 原始图像坐标信息 (用于坐标转换)
        vggt_resolution: VGGT 内部分辨率 (默认 518)
        point_size: 点的大小
        max_points: 最大显示点数（避免过多点导致图像混乱）
    
    Returns:
        vis_image: 可视化后的图像 (BGR 格式，用于 OpenCV)
    """
    # 加载原始图像
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法加载图像: {image_path}")
    
    img_h, img_w = image.shape[:2]
    vis_image = image.copy()
    
    if tracks.shape[0] == 0:
        return vis_image
    
    # 限制显示的轨迹点数
    if tracks.shape[0] > max_points:
        indices = np.random.choice(tracks.shape[0], max_points, replace=False)
        tracks = tracks[indices]
        if vis_scores is not None:
            vis_scores = vis_scores[indices]
        if confs is not None:
            confs = confs[indices]
    
    # 坐标转换：从 VGGT 分辨率 (518x518) 转换到原始图像分辨率
    # original_coords 格式: [x1, y1, x2, y2, width, height]
    # 其中 x1, y1, x2, y2 是在加载分辨率（如 768x768）下的坐标
    if original_coords is not None:
        try:
            # 转换为 numpy 数组（如果是 torch.Tensor）
            coords = original_coords
            if hasattr(coords, 'cpu'):
                coords = coords.cpu().numpy()
            elif hasattr(coords, 'numpy'):
                coords = coords.numpy()
            
            # 确保是 1D 数组，如果不是则尝试展平
            if hasattr(coords, 'ndim'):
                if coords.ndim > 1:
                    coords = coords.flatten()
                elif coords.ndim == 0:
                    # 如果是标量，无法使用，跳过转换
                    coords = None
            elif not isinstance(coords, (list, tuple, np.ndarray)):
                # 如果不是数组类型，设为 None
                coords = None
        except (AttributeError, TypeError) as e:
            # 如果转换失败，设为 None
            coords = None
        
        # --- 修复后的代码段 ---
        if coords is not None:
            try:
                # 检查 coords 是否是可以索引的序列（列表或数组）
                if hasattr(coords, "__getitem__") and not np.isscalar(coords):
                    x1 = float(coords[0])
                    y1 = float(coords[1])
                    x2 = float(coords[2])
                    y2 = float(coords[3])
                    original_w = float(coords[4])
                    original_h = float(coords[5])
                    
                    scale_to_original = original_w / (x2 - x1)
                    tracks_converted = tracks.copy().astype(np.float32)
                    tracks_converted[:, 0] = (tracks_converted[:, 0] - x1) * scale_to_original
                    tracks_converted[:, 1] = (tracks_converted[:, 1] - y1) * scale_to_original
                    tracks = tracks_converted
                else:
                    # 如果是标量或不可索引，回退到简单缩放
                    scale = img_w / vggt_resolution
                    tracks = tracks * scale
            except (IndexError, TypeError):
                scale = img_w / vggt_resolution
                tracks = tracks * scale
        # --- 修复结束 ---
    else:
        # 如果没有 original_coords，简单缩放（不准确）
        scale = img_w / vggt_resolution
        tracks = tracks * scale
    
    # 根据可见性分数设置颜色（绿色=高可见性，红色=低可见性）
    N = tracks.shape[0]
    if vis_scores is not None:
        colors = np.zeros((N, 3), dtype=np.uint8)
        for i in range(N):
            vis = float(vis_scores[i])
            # BGR 格式：绿色到红色的渐变
            colors[i] = [0, int(vis * 255), int((1 - vis) * 255)]
    elif confs is not None:
        colors = np.zeros((N, 3), dtype=np.uint8)
        conf_min, conf_max = float(confs.min()), float(confs.max())
        if conf_max > conf_min:
            conf_normalized = (confs - conf_min) / (conf_max - conf_min)
        else:
            conf_normalized = np.ones(N)
        for i in range(N):
            conf_norm = float(conf_normalized[i])
            # BGR 格式：蓝色到红色的渐变
            colors[i] = [int((1 - conf_norm) * 255), 0, int(conf_norm * 255)]
    else:
        # 固定颜色（黄色）
        colors = np.array([[0, 255, 255]] * N, dtype=np.uint8)  # 黄色 (BGR)
    
    # 绘制轨迹点
    for i in range(N):
        x, y = float(tracks[i, 0]), float(tracks[i, 1])
        # 确保坐标在图像范围内
        x = int(np.clip(x, 0, img_w - 1))
        y = int(np.clip(y, 0, img_h - 1))
        color = tuple(map(int, colors[i]))
        cv2.circle(vis_image, (x, y), point_size, color, -1)
    
    return vis_image


def create_track_comparison_image(original_image_path, tracks, vis_scores=None, confs=None,
                                  original_coords=None, vggt_resolution=518,
                                  point_size=3, max_points=2000):
    """
    创建原图与轨迹图的对比图（并排显示）
    
    Args:
        original_image_path: 原始图像路径
        tracks: 轨迹点坐标 [N, 2]
        vis_scores: 可见性分数 [N] (可选)
        confs: 置信度 [N] (可选)
        original_coords: 原始图像坐标信息
        vggt_resolution: VGGT 内部分辨率
        point_size: 点的大小
        max_points: 最大显示点数
    
    Returns:
        comparison_image: 对比图像 (BGR 格式)
    """
    # 加载原始图像
    original_image = cv2.imread(original_image_path)
    if original_image is None:
        raise ValueError(f"无法加载图像: {original_image_path}")
    
    # 创建带轨迹点的图像
    tracks_image = visualize_tracks_on_image(
        original_image_path, tracks, vis_scores, confs,
        original_coords, vggt_resolution, point_size, max_points=max_points
    )
    
    # 并排拼接
    comparison_image = np.hstack([original_image, tracks_image])
    
    # 添加文字标签
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 2
    color = (255, 255, 255)  # 白色
    
    # 在左侧图像上添加"原图"标签
    cv2.putText(comparison_image, "Original", (20, 40), 
                font, font_scale, color, thickness)
    
    # 在右侧图像上添加"轨迹图"标签
    cv2.putText(comparison_image, "Tracks", (original_image.shape[1] + 20, 40), 
                font, font_scale, color, thickness)
    
    return comparison_image


def main():
    args = parse_args()
    
    # 设置输出目录
    if args.output_dir is None:
        args.output_dir = args.scene_dir
    
    # 固定使用 GPU
    device = setup_device()
    
    # 在加载模型前，先检查并清理 GPU 内存
    if device == "cuda":
        # 检查是否有其他进程占用 GPU
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        print(f"GPU 内存状态: 已分配 {allocated:.2f} GB, 已保留 {reserved:.2f} GB")
        
        # 清理所有缓存
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        print("✓ GPU 缓存已清理")
    
    # 加载模型
    model = load_model(device)
    
    # 获取图像路径
    image_dir = os.path.join(args.scene_dir, "images")
    if not os.path.exists(image_dir):
        raise ValueError(f"图像目录不存在: {image_dir}")
    
    import glob
    image_paths = sorted(glob.glob(os.path.join(image_dir, "*")))
    image_paths = [p for p in image_paths if p.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if len(image_paths) == 0:
        raise ValueError(f"在 {image_dir} 中未找到图片文件")
    
    # 处理指定帧数（默认 8）
    MAX_FRAMES = args.num_frames
    if len(image_paths) > MAX_FRAMES:
        # 等距采样，覆盖整个序列
        indices = np.linspace(0, len(image_paths) - 1, MAX_FRAMES).astype(int)
        image_paths = [image_paths[i] for i in indices]

    print(f"处理 {len(image_paths)} 张图片")
    
    # 加载图像（降低分辨率以节省显存）
    print("加载图像...")
    img_load_resolution = 518  # 降低分辨率以适配 12GB GPU
    with tqdm(total=len(image_paths), desc="加载图像", unit="张") as pbar:
        images, original_coords = load_and_preprocess_images_square(image_paths, img_load_resolution)
        pbar.update(len(image_paths))
    images = images.to(device)
    original_coords = original_coords.to(device)
    print(f"图像加载完成: {images.shape} (分辨率: {img_load_resolution})")
    
    # 清理显存
    torch.cuda.empty_cache()
    
    # 运行 VGGT 推理
    vggt_resolution = 518  # VGGT 内部固定分辨率
    print(f"运行 VGGT 推理（内部分辨率: {vggt_resolution}）...")
    extrinsic, intrinsic, depth_map, depth_conf = run_vggt_inference(model, images, device, vggt_resolution)

    # 释放模型，回收显存
    del model
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    
    # 计算 3D 点
    print("计算 3D 世界坐标点...")
    with tqdm(total=1, desc="计算 3D 点", unit="批次") as pbar:
        world_points = unproject_depth_map_to_point_map(depth_map, extrinsic, intrinsic)
        pbar.update(1)
    print(f"✓ 3D 点计算完成: {world_points.shape}")
    
    # 清理显存
    torch.cuda.empty_cache()
    
    # 计算轨迹（固定参数，5张图片）
    # 检查 GPU 剩余内存，如果不足则跳过轨迹计算
    torch.cuda.empty_cache()
    gpu_free = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / 1024**3
    skip_tracks = gpu_free < 4.0
    if skip_tracks:
        print(f"GPU free memory {gpu_free:.1f} GB < 4 GB, skipping track computation")
        pred_tracks = None
        pred_vis_scores = None
        pred_confs = None

    if not skip_tracks:
        print("计算轨迹...")
        try:
            from vggt.dependency.track_predict import predict_tracks
        except ImportError as e:
            print(f"Cannot import predict_tracks: {e}")
            pred_tracks = None
            pred_vis_scores = None
            pred_confs = None
            skip_tracks = True

    if not skip_tracks:
        # 固定参数（5张图片，使用保守参数避免OOM）
        max_query_pts = 256
        query_frame_num = 2
        fine_tracking = False
        max_points_num = 10240
        print(f"轨迹计算参数: max_query_pts={max_query_pts}, query_frame_num={query_frame_num}, max_points_num={max_points_num}")
        
        # 准备数据
        track_images = images
        track_conf = torch.from_numpy(depth_conf).to(device)
        track_points_3d = torch.from_numpy(world_points).to(device)
        
        # 清理显存碎片（在轨迹计算前释放更多内存）
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # 如果 GPU 内存仍然紧张，尝试释放 VGGT 模型占用的内存
        # 注意：这可能会影响后续操作，但轨迹计算不需要 VGGT 模型
        if device == "cuda":
            # 检查 GPU 内存使用情况
            allocated = torch.cuda.memory_allocated(0) / 1024**3  # GB
            reserved = torch.cuda.memory_reserved(0) / 1024**3  # GB
            print(f"GPU 内存使用: 已分配 {allocated:.2f} GB, 已保留 {reserved:.2f} GB")
            
            # 如果内存使用超过阈值，尝试释放更多
            if allocated > 9.0:  # 假设 12GB GPU，超过 9GB 就警告
                print("⚠️  GPU 内存使用较高，尝试释放更多内存...")
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        
        # 使用 autocast
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        use_autocast = True
        
        with tqdm(total=1, desc="轨迹计算", unit="批次", bar_format='{desc}: {percentage:3.0f}%|{bar}| 预计时间较长...') as pbar:
            if use_autocast:
                autocast_context = torch.cuda.amp.autocast(dtype=dtype, enabled=True)
                with autocast_context:
                    pred_tracks, pred_vis_scores, pred_confs, points_3d, points_rgb = predict_tracks(
                        track_images,
                        conf=track_conf,
                        points_3d=track_points_3d,
                        masks=None,
                        max_query_pts=max_query_pts,
                        query_frame_num=query_frame_num,
                        keypoint_extractor="aliked+sp",
                        max_points_num=max_points_num,
                        fine_tracking=fine_tracking,
                        complete_non_vis=True,
                    )
            else:
                pred_tracks, pred_vis_scores, pred_confs, points_3d, points_rgb = predict_tracks(
                    track_images,
                    conf=track_conf,
                    points_3d=track_points_3d,
                    masks=None,
                    max_query_pts=max_query_pts,
                    query_frame_num=query_frame_num,
                    keypoint_extractor="aliked+sp",
                    max_points_num=max_points_num,
                    fine_tracking=fine_tracking,
                    complete_non_vis=True,
                )
            pbar.update(1)
        
        # 清理显存
        del track_images, track_conf, track_points_3d
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print(f"✓ 轨迹计算完成: {pred_tracks.shape}")
    
    # 创建输出目录
    vggt_output_dir = os.path.join(args.output_dir, "vggt")
    os.makedirs(vggt_output_dir, exist_ok=True)

    # 保存 W_geo 配置信息（用于消融实验追溯）
    wgeo_config = {
        "w_sigma": args.w_sigma,
        "w_vis": args.w_vis,
        "w_cons": args.w_cons,
        "strategy": args.w_strategy,
        "min_weight": args.w_min,
        "num_frames": args.num_frames,
    }
    import json as _json
    with open(os.path.join(vggt_output_dir, "wgeo_config.json"), 'w') as f:
        _json.dump(wgeo_config, f, indent=2)
    print(f"[OK] W_geo config saved: {wgeo_config}")

    # 处理每一帧
    S = depth_map.shape[0]
    frame_list = []
    
    # 计算可见性（如果有轨迹）
    vis_full = None
    if pred_vis_scores is not None:
        print("计算全图可见性（从轨迹插值）...")
        vis_full = compute_visibility_from_tracks(pred_vis_scores, pred_tracks, depth_map.shape[1:3])
        print(f"✓ 可见性计算完成: {vis_full.shape}")
    
    print("处理每一帧数据并保存...")
    for frame_idx in tqdm(range(S), desc="处理帧", unit="帧"):
        frame_name = f"frame_{frame_idx:03d}"
        frame_list.append(frame_name)
        
        # 获取当前帧数据
        depth = depth_map[frame_idx].squeeze()  # [H, W]
        # depth_conf 已经在 CPU 上（numpy），直接使用
        conf = depth_conf[frame_idx]  # [H, W]
        extri = extrinsic[frame_idx]  # [3, 4]
        intri = intrinsic[frame_idx]  # [3, 3]
        original_size = original_coords[frame_idx].cpu().numpy()[-2:]  # [W, H]
        
        # 计算不确定度
        sigma = compute_uncertainty(conf)
        
        # 获取可见性
        if vis_full is not None:
            vis = vis_full[frame_idx]
        else:
            # 使用全 1 占位
            vis = np.ones_like(sigma)
        
        # 计算一致性（如果有轨迹）
        consistency = None
        if pred_tracks is not None:
            consistency_scores = compute_consistency_from_tracks(
                world_points, pred_tracks, extrinsic, intrinsic
            )
            # 用与可见性相同的插值方法，把稀疏轨迹点分数扩散到全图
            consistency_full = compute_visibility_from_tracks(
                consistency_scores, pred_tracks, depth_map.shape[1:3]
            )
            consistency = consistency_full[frame_idx]  # [H, W]
        
        # 归一化信号到 0-1（越大越可信）
        sigma_reliability = 1.0 - sigma  # 不确定度反向
        vis_reliability = percentile_normalize(vis)  # 可见性归一化
        consistency_reliability = percentile_normalize(consistency) if consistency is not None else None
        
        # 计算权重图（参数从命令行传入，支持消融实验）
        W = compute_weight_map(
            sigma_reliability, vis_reliability, consistency_reliability,
            w_sigma=args.w_sigma, w_vis=args.w_vis, w_cons=args.w_cons,
            strategy=args.w_strategy, min_weight=args.w_min
        )
        
        # 保存数据
        # 1. 相机参数 JSON
        original_image_name = os.path.basename(image_paths[frame_idx])
        save_camera_params(
            frame_idx, extri, intri, depth.shape, 
            os.path.join(vggt_output_dir, f"{frame_name}.json"),
            f"images/{original_image_name}"
        )
        
        # 2. 深度图 NPY
        np.save(os.path.join(vggt_output_dir, f"{frame_name}_depth.npy"), depth)
        
        # 3. 不确定度 NPY
        np.save(os.path.join(vggt_output_dir, f"{frame_name}_sigma.npy"), sigma)
        
        # 4. 可见性 NPY
        np.save(os.path.join(vggt_output_dir, f"{frame_name}_vis.npy"), vis)

        # 4b. 一致性 NPY（保存以支持 W_geo 系数消融实验，无需重跑 VGGT）
        if consistency is not None:
            np.save(os.path.join(vggt_output_dir, f"{frame_name}_cons.npy"), consistency)

        # 5. 权重图 NPY
        np.save(os.path.join(vggt_output_dir, f"{frame_name}_W.npy"), W)
        
        # 6. 可视化预览图
        create_visualization(depth, sigma, W, vggt_output_dir, frame_name)
    
    # 保存轨迹（如果有）
    if pred_tracks is not None:
        print("保存轨迹数据...")
        # points_3d 是所有查询帧合并后的 [N_total, 3]，不是按帧分开的
        # 所以需要保存整个数组，而不是 points_3d[frame_idx]
        for frame_idx in tqdm(range(S), desc="保存轨迹", unit="帧", leave=False):
            frame_name = f"frame_{frame_idx:03d}"
            save_dict = {
                "tracks": pred_tracks[frame_idx],  # [N, 2] - 当前帧的轨迹点
            }
            if pred_vis_scores is not None:
                save_dict["vis_scores"] = pred_vis_scores[frame_idx]  # [N] - 当前帧的可见性
            if pred_confs is not None:
                save_dict["confs"] = pred_confs[frame_idx] if pred_confs.ndim > 1 else pred_confs
            # 保存轨迹点对应的 3D 点
            # 注意：pred_tracks 是 [S, N, 2]，points_3d 是 [N, 3]
            # 问题：points_3d 来自不同查询帧的坐标系，不能直接用于当前帧的投影验证
            # 解决方案：使用当前帧的深度图反投影获取 3D 点（在正确的坐标系下）
            if points_3d is not None and depth_map is not None:
                # 从当前帧的深度图反投影获取 3D 点（使用当前帧的外参和内参）
                from vggt.utils.geometry import depth_to_world_coords_points
                current_depth = depth_map[frame_idx].squeeze()  # [H, W]，与第667行保持一致
                current_world_points, _, _ = depth_to_world_coords_points(
                    current_depth,
                    extri,  # 使用当前帧的外参（已在第669行获取）
                    intri   # 使用当前帧的内参（已在第670行获取）
                )  # [H, W, 3]
                
                # 获取当前帧的轨迹点
                current_tracks = pred_tracks[frame_idx]  # [N, 2]
                n_tracks = current_tracks.shape[0]
                
                # 使用双线性插值获取轨迹点位置的 3D 点
                try:
                    from scipy.interpolate import griddata
                    H, W = current_world_points.shape[:2]
                    y_grid, x_grid = np.mgrid[0:H, 0:W]
                    points_grid = np.column_stack([x_grid.ravel(), y_grid.ravel()])
                    world_points_flat = current_world_points.reshape(-1, 3)
                    
                    # 双线性插值
                    points_3d_current_frame = griddata(
                        points_grid, world_points_flat,
                        current_tracks, method='linear', fill_value=np.nan
                    )  # [N, 3]
                    
                    # 检查插值结果
                    valid_mask = ~np.isnan(points_3d_current_frame).any(axis=1)
                    if valid_mask.sum() < n_tracks * 0.9:
                        # 对于插值失败的点，使用最近邻插值作为后备
                        points_3d_fallback = griddata(
                            points_grid, world_points_flat,
                            current_tracks[~valid_mask], method='nearest', fill_value=0
                        )
                        points_3d_current_frame[~valid_mask] = points_3d_fallback
                    
                    # 保存当前帧坐标系下的 3D 点
                    save_dict["points_3d"] = points_3d_current_frame  # [N, 3]
                except ImportError:
                    # 如果没有 scipy，使用直接索引（精度较低）
                    print(f"⚠️  警告: 未安装 scipy，使用直接索引获取 3D 点（精度较低）")
                    track_indices = current_tracks.astype(int)
                    H, W = current_world_points.shape[:2]
                    track_indices[:, 0] = np.clip(track_indices[:, 0], 0, W-1)
                    track_indices[:, 1] = np.clip(track_indices[:, 1], 0, H-1)
                    points_3d_current_frame = current_world_points[track_indices[:, 1], track_indices[:, 0]]  # [N, 3]
                    save_dict["points_3d"] = points_3d_current_frame
            elif points_3d is not None:
                # 如果没有深度图，回退到使用原始的 points_3d（但会有坐标系不匹配的问题）
                print(f"⚠️  警告: 帧 {frame_idx} 没有深度图，使用原始 points_3d（可能坐标系不匹配）")
                if points_3d.ndim == 2:
                    n_tracks = pred_tracks[frame_idx].shape[0]
                    n_points_3d = points_3d.shape[0]
                    if n_tracks == n_points_3d:
                        save_dict["points_3d"] = points_3d
                    else:
                        min_n = min(n_tracks, n_points_3d)
                        save_dict["points_3d"] = points_3d[:min_n]
                        if n_tracks != n_points_3d:
                            print(f"⚠️  警告: 帧 {frame_idx} 轨迹点数量 ({n_tracks}) 与 3D 点数量 ({n_points_3d}) 不匹配")
            np.savez(
                os.path.join(vggt_output_dir, f"{frame_name}_tracks.npz"),
                **save_dict
            )
            
            # 生成轨迹可视化图（原图 + 轨迹点对比）
            try:
                original_image_path = image_paths[frame_idx]
                current_tracks = pred_tracks[frame_idx]  # [N, 2]
                current_vis_scores = pred_vis_scores[frame_idx] if pred_vis_scores is not None else None
                current_confs = pred_confs[frame_idx] if pred_confs is not None else None
                
                # 正确提取 original_coords（处理 torch.Tensor）
                if original_coords is not None:
                    try:
                        # 先获取指定帧的坐标
                        if hasattr(original_coords, 'cpu'):
                            # torch.Tensor，需要先移到 CPU 再转换为 numpy
                            frame_coords = original_coords[frame_idx]
                            if hasattr(frame_coords, 'cpu'):
                                current_original_coords = frame_coords.cpu().numpy()
                            else:
                                current_original_coords = frame_coords.numpy() if hasattr(frame_coords, 'numpy') else frame_coords
                        elif hasattr(original_coords, 'numpy'):
                            # 已经是 numpy 兼容格式
                            current_original_coords = original_coords[frame_idx].numpy()
                        else:
                            # 已经是 numpy 数组
                            current_original_coords = original_coords[frame_idx]
                        
                        # 确保是 1D 数组
                        if hasattr(current_original_coords, 'ndim'):
                            if current_original_coords.ndim == 0:
                                # 如果是标量，说明格式不对，设为 None
                                current_original_coords = None
                            elif current_original_coords.ndim > 1:
                                # 如果是多维数组，展平
                                current_original_coords = current_original_coords.flatten()
                        elif not isinstance(current_original_coords, (list, tuple, np.ndarray)):
                            # 如果不是数组类型，设为 None
                            current_original_coords = None
                    except (IndexError, TypeError, AttributeError) as e:
                        # 如果索引失败，设为 None
                        print(f"⚠️  警告: 无法提取帧 {frame_idx} 的 original_coords: {e}")
                        current_original_coords = None
                else:
                    current_original_coords = None
                
                # 创建对比图
                comparison_image = create_track_comparison_image(
                    original_image_path,
                    current_tracks,
                    vis_scores=current_vis_scores,
                    confs=current_confs,
                    original_coords=current_original_coords,
                    vggt_resolution=vggt_resolution,
                    point_size=3,
                    max_points=2000
                )
                
                # 保存对比图
                output_path = os.path.join(vggt_output_dir, f"{frame_name}_tracks_comparison.png")
                cv2.imwrite(output_path, comparison_image)
                
            except Exception as e:
                print(f"⚠️  警告: 帧 {frame_idx} 轨迹可视化失败: {e}")
    
    # 保存场景元信息
    scene_meta = {
        "scene_name": os.path.basename(args.scene_dir),
        "num_frames": S,
        "frame_list": frame_list,
        "image_size": [int(depth_map.shape[1]), int(depth_map.shape[2])],
        "vggt_resolution": vggt_resolution,
        "coordinate_normalization": {
            "world_frame": "first_frame",
            "depth_scale": "normalized",
            "coordinate_system": "OpenCV"
        },
        "fusion_strategy": "multiplicative",
        "fusion_weights": {
            "uncertainty": 0.5,
            "visibility": 0.3,
            "consistency": 0.2
        },
        "has_tracks": pred_tracks is not None,
        "has_visibility": vis_full is not None
    }
    
    print("保存场景元信息...")
    with open(os.path.join(args.output_dir, "scene_meta.json"), 'w') as f:
        json.dump(scene_meta, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"✓ 处理完成！")
    print(f"  结果保存在: {vggt_output_dir}")
    print(f"  场景元信息: {os.path.join(args.output_dir, 'scene_meta.json')}")
    print(f"  处理了 {S} 帧图像")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

