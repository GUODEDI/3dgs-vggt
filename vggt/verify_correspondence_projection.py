#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对应点投影验证脚本

用于验证 3D 世界坐标点投影到 2D 图像坐标的正确性，
检查是否存在坐标系转换（C2W/W2C）错误。

用法:
    python verify_correspondence_projection.py --vggt_dir examples/kitchen/vggt --frame_idx 0
"""

import os
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import json

from vggt.dependency.projection import project_3D_points_np
from vggt.utils.geometry import closed_form_inverse_se3


def load_vggt_data(vggt_dir, frame_idx):
    """加载 VGGT 输出数据"""
    vggt_dir = Path(vggt_dir)
    
    # 加载轨迹数据
    tracks_file = vggt_dir / f"frame_{frame_idx:03d}_tracks.npz"
    if not tracks_file.exists():
        raise FileNotFoundError(f"轨迹文件不存在: {tracks_file}")
    
    tracks_data = np.load(tracks_file)
    pred_tracks = tracks_data["tracks"]  # [N, 2]
    vis_scores = tracks_data.get("vis_scores", None)
    pred_points_3d = tracks_data.get("points_3d", None)  # [N, 3] - 轨迹点对应的 3D 点
    
    # 加载相机参数
    camera_file = vggt_dir / f"frame_{frame_idx:03d}.json"
    if not camera_file.exists():
        raise FileNotFoundError(f"相机参数文件不存在: {camera_file}")
    
    with open(camera_file, 'r') as f:
        camera_data = json.load(f)
    
    extrinsic = np.array(camera_data["extrinsic"]["matrix"])  # [3, 4]
    intrinsic = np.array(camera_data["intrinsic"]["matrix"])  # [3, 3]
    
    # 加载深度图（如果存在）
    depth_file = vggt_dir / f"frame_{frame_idx:03d}_depth.npy"
    depth_map = None
    if depth_file.exists():
        depth_map = np.load(depth_file)
    
    return {
        "pred_tracks": pred_tracks,
        "vis_scores": vis_scores,
        "pred_points_3d": pred_points_3d,  # 轨迹点对应的 3D 点
        "extrinsic": extrinsic,
        "intrinsic": intrinsic,
        "depth_map": depth_map,
    }


def verify_extrinsic_matrix(extrinsic):
    """验证外参矩阵的有效性"""
    print("\n" + "="*60)
    print("外参矩阵验证")
    print("="*60)
    
    R = extrinsic[:3, :3]
    t = extrinsic[:3, 3]
    
    # 1. 检查旋转矩阵的行列式
    det = np.linalg.det(R)
    print(f"旋转矩阵行列式: {det:.6f} (应该接近 1.0)")
    if abs(det - 1.0) > 0.01:
        print("  ⚠️  警告: 旋转矩阵行列式偏离 1.0，可能不是有效的旋转矩阵")
    
    # 2. 检查是否是正交矩阵
    RRT = R @ R.T
    is_orthogonal = np.allclose(RRT, np.eye(3), atol=1e-5)
    print(f"旋转矩阵是否正交: {is_orthogonal}")
    if not is_orthogonal:
        print("  ⚠️  警告: 旋转矩阵不是正交矩阵")
        print(f"  R @ R.T 的最大偏差: {np.abs(RRT - np.eye(3)).max():.6f}")
    
    # 3. 检查外参矩阵格式（W2C vs C2W）
    # 如果外参是 W2C，那么相机在世界坐标系中的位置应该是 -R^T @ t
    # 如果外参是 C2W，那么相机位置应该是 t
    # 这里我们假设是 W2C，计算相机位置
    camera_pos_w2c = -R.T @ t
    print(f"\n假设外参是 W2C (World-to-Camera):")
    print(f"  相机在世界坐标系中的位置: [{camera_pos_w2c[0]:.3f}, {camera_pos_w2c[1]:.3f}, {camera_pos_w2c[2]:.3f}]")
    
    # 如果外参是 C2W，相机位置就是 t
    print(f"\n假设外参是 C2W (Camera-to-World):")
    print(f"  相机在世界坐标系中的位置: [{t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}]")
    
    return {
        "det": det,
        "is_orthogonal": is_orthogonal,
        "camera_pos_w2c": camera_pos_w2c,
        "camera_pos_c2w": t,
    }


def verify_projection_from_points_3d(pred_tracks, pred_points_3d, extrinsic, intrinsic):
    """使用轨迹点对应的 3D 点进行验证（推荐方法）"""
    print("\n" + "="*60)
    print("从轨迹点对应的 3D 点验证投影")
    print("="*60)
    
    # 检查并处理 pred_points_3d 的形状
    pred_points_3d = np.asarray(pred_points_3d)
    print(f"pred_points_3d 原始形状: {pred_points_3d.shape}")
    print(f"pred_tracks 形状: {pred_tracks.shape}")
    
    # 处理不同的形状
    if pred_points_3d.ndim == 1:
        # 如果是 1D，尝试重塑
        if pred_points_3d.shape[0] % 3 == 0:
            pred_points_3d = pred_points_3d.reshape(-1, 3)
            print(f"  重塑为: {pred_points_3d.shape}")
        else:
            print(f"⚠️  错误: pred_points_3d 是 1D 数组但无法重塑为 [N, 3]，形状: {pred_points_3d.shape}")
            return None
    elif pred_points_3d.ndim == 2:
        # 期望的形状: [N, 3]
        if pred_points_3d.shape[1] != 3:
            # 尝试转置
            if pred_points_3d.shape[0] == 3:
                pred_points_3d = pred_points_3d.T
                print(f"  转置为: {pred_points_3d.shape}")
            else:
                print(f"⚠️  错误: pred_points_3d 形状不正确: {pred_points_3d.shape}，期望 [N, 3]")
                return None
    elif pred_points_3d.ndim == 3:
        # 可能是 [1, N, 3] 或 [S, N, 3]
        if pred_points_3d.shape[0] == 1:
            pred_points_3d = pred_points_3d[0]
            print(f"  去除第一个维度: {pred_points_3d.shape}")
        elif pred_points_3d.shape[2] == 3:
            # 可能是 [S, N, 3]，取第一帧
            pred_points_3d = pred_points_3d[0]
            print(f"  取第一帧: {pred_points_3d.shape}")
        else:
            print(f"⚠️  错误: pred_points_3d 3D 形状不正确: {pred_points_3d.shape}")
            return None
    else:
        print(f"⚠️  错误: pred_points_3d 维度不正确: {pred_points_3d.ndim}，期望 2D")
        return None
    
    # 最终检查
    if pred_points_3d.ndim != 2 or pred_points_3d.shape[1] != 3:
        print(f"⚠️  错误: 处理后 pred_points_3d 形状仍不正确: {pred_points_3d.shape}")
        return None
    
    # pred_tracks 形状: [N, 2]
    # pred_points_3d 形状: [N, 3]
    N = pred_tracks.shape[0]
    N_3d = pred_points_3d.shape[0]
    
    if N != N_3d:
        print(f"⚠️  警告: 轨迹点数量 ({N}) 与 3D 点数量 ({N_3d}) 不匹配")
        # 取较小的数量
        min_N = min(N, N_3d)
        pred_tracks = pred_tracks[:min_N]
        pred_points_3d = pred_points_3d[:min_N]
        N = min_N
        print(f"  使用前 {N} 个点进行验证")
    
    # 确保 pred_points_3d 是 [N, 3] 形状
    if pred_points_3d.shape != (N, 3):
        pred_points_3d = pred_points_3d.reshape(-1, 3)[:N]
    
    # 将 3D 点投影回 2D
    # 注意：pred_points_3d 可能来自查询帧的坐标系
    # 如果查询帧和当前帧不同，可能需要使用查询帧的外参
    # 但这里先假设 pred_points_3d 在当前帧的坐标系下
    
    # 调试：检查 3D 点的范围
    print(f"3D 点范围:")
    print(f"  X: [{pred_points_3d[:, 0].min():.3f}, {pred_points_3d[:, 0].max():.3f}]")
    print(f"  Y: [{pred_points_3d[:, 1].min():.3f}, {pred_points_3d[:, 1].max():.3f}]")
    print(f"  Z: [{pred_points_3d[:, 2].min():.3f}, {pred_points_3d[:, 2].max():.3f}]")
    
    # 检查是否有异常值
    z_positive = (pred_points_3d[:, 2] > 0).sum()
    z_negative = (pred_points_3d[:, 2] < 0).sum()
    print(f"  Z > 0 的点: {z_positive} ({z_positive/N*100:.1f}%)")
    print(f"  Z < 0 的点: {z_negative} ({z_negative/N*100:.1f}%)")
    if z_negative > z_positive:
        print(f"  ⚠️  警告: 大部分点的 Z < 0，可能坐标系有问题（相机坐标系 Z 应该向前）")
    
    # 检查内参矩阵
    print(f"\n内参矩阵:")
    print(f"  fx={intrinsic[0, 0]:.1f}, fy={intrinsic[1, 1]:.1f}")
    print(f"  cx={intrinsic[0, 2]:.1f}, cy={intrinsic[1, 2]:.1f}")
    
    # 检查外参矩阵的平移部分
    print(f"\n外参矩阵平移部分: [{extrinsic[0, 3]:.3f}, {extrinsic[1, 3]:.3f}, {extrinsic[2, 3]:.3f}]")
    
    # 测试：将 3D 点转换到相机坐标系
    from vggt.utils.geometry import closed_form_inverse_se3
    # 如果外参是 W2C，直接使用；如果是 C2W，需要求逆
    # 先假设是 W2C，计算相机坐标系下的点
    points_cam_test = (extrinsic @ np.concatenate([pred_points_3d[:5], np.ones((5, 1))], axis=1).T).T
    print(f"\n测试：前 5 个点在相机坐标系下的 Z 值:")
    print(f"  {points_cam_test[:, 2]}")
    if (points_cam_test[:, 2] < 0).all():
        print(f"  ⚠️  警告: 所有测试点的 Z < 0，可能外参矩阵格式错误（应该是 C2W 而不是 W2C）")
    
    # 测试：尝试使用外参的逆矩阵（如果是 C2W 而不是 W2C）
    from vggt.utils.geometry import closed_form_inverse_se3
    
    # 方法 1: 假设外参是 W2C（当前方法）
    projected_points_w2c, points_cam_w2c = project_3D_points_np(
        pred_points_3d,  # [N, 3]
        extrinsic[None], # [1, 3, 4]
        intrinsic[None]  # [1, 3, 3]
    )
    projected_points_w2c = projected_points_w2c[0]  # [N, 2]
    points_cam_w2c = points_cam_w2c[0]  # [3, N]
    
    # 方法 2: 尝试使用外参的逆矩阵（如果是 C2W）
    extrinsic_4x4 = np.eye(4)
    extrinsic_4x4[:3, :] = extrinsic
    extrinsic_c2w = closed_form_inverse_se3(extrinsic_4x4[None])[0, :3, :]  # [3, 4] C2W
    
    projected_points_c2w, points_cam_c2w = project_3D_points_np(
        pred_points_3d,  # [N, 3]
        extrinsic_c2w[None], # [1, 3, 4] C2W（错误用法，但用于测试）
        intrinsic[None]  # [1, 3, 3]
    )
    projected_points_c2w = projected_points_c2w[0]  # [N, 2]
    points_cam_c2w = points_cam_c2w[0]  # [3, N]
    
    # 比较两种方法的结果
    print(f"\n方法对比（前 5 个点）:")
    print(f"  假设 W2C - 投影点 X: {projected_points_w2c[:5, 0]}")
    print(f"  假设 C2W - 投影点 X: {projected_points_c2w[:5, 0]}")
    print(f"  轨迹点 X: {pred_tracks[:5, 0]}")
    
    # 选择更接近轨迹点的方法
    error_w2c = np.linalg.norm(projected_points_w2c - pred_tracks, axis=1).mean()
    error_c2w = np.linalg.norm(projected_points_c2w - pred_tracks, axis=1).mean()
    
    print(f"\n重投影误差对比:")
    print(f"  假设 W2C: {error_w2c:.2f} 像素")
    print(f"  假设 C2W: {error_c2w:.2f} 像素")
    
    if error_c2w < error_w2c:
        print(f"  ⚠️  警告: 使用 C2W 假设误差更小，可能外参矩阵实际上是 C2W 格式")
        print(f"  但这是错误的用法（C2W 不能直接用于投影），说明坐标系有问题")
        # 仍然使用 W2C 方法，但记录问题
        projected_points = projected_points_w2c
        points_cam = points_cam_w2c
    else:
        projected_points = projected_points_w2c
        points_cam = points_cam_w2c
    
    # 调试：检查相机坐标系下的点
    print(f"\n相机坐标系下的点（前 5 个）:")
    print(f"  X_cam: {points_cam[0, :5]}")
    print(f"  Y_cam: {points_cam[1, :5]}")
    print(f"  Z_cam: {points_cam[2, :5]}")
    
    # 检查归一化坐标（除以 Z 后）
    normalized_xy = points_cam[:2, :] / (points_cam[2:3, :] + 1e-8)
    print(f"归一化坐标（X/Z, Y/Z）范围:")
    print(f"  X/Z: [{normalized_xy[0].min():.3f}, {normalized_xy[0].max():.3f}]")
    print(f"  Y/Z: [{normalized_xy[1].min():.3f}, {normalized_xy[1].max():.3f}]")
    
    # 调试：检查投影点的范围
    print(f"\n投影点范围:")
    print(f"  X: [{projected_points[:, 0].min():.1f}, {projected_points[:, 0].max():.1f}] (图像宽度: 518)")
    print(f"  Y: [{projected_points[:, 1].min():.1f}, {projected_points[:, 1].max():.1f}] (图像高度: 518)")
    
    # 检查是否有异常大的投影值
    out_of_bounds = ((projected_points[:, 0] < -1000) | (projected_points[:, 0] > 2000) | 
                     (projected_points[:, 1] < -1000) | (projected_points[:, 1] > 2000)).sum()
    if out_of_bounds > 0:
        print(f"  ⚠️  警告: {out_of_bounds} 个点的投影坐标异常大（超出合理范围）")
    
    # 检查内参是否合理
    if intrinsic[0, 0] < 10 or intrinsic[0, 0] > 10000:
        print(f"  ⚠️  警告: 内参 fx={intrinsic[0, 0]:.1f} 看起来不合理（应该在 100-1000 范围内）")
    if intrinsic[1, 1] < 10 or intrinsic[1, 1] > 10000:
        print(f"  ⚠️  警告: 内参 fy={intrinsic[1, 1]:.1f} 看起来不合理（应该在 100-1000 范围内）")
    
    # 计算重投影误差
    reproj_errors = np.linalg.norm(projected_points - pred_tracks, axis=1)
    
    print(f"轨迹点数量: {N}")
    print(f"重投影误差统计:")
    print(f"  平均误差: {reproj_errors.mean():.3f} 像素")
    print(f"  中位数误差: {np.median(reproj_errors):.3f} 像素")
    print(f"  最大误差: {reproj_errors.max():.3f} 像素")
    print(f"  最小误差: {reproj_errors.min():.3f} 像素")
    print(f"  误差 < 1 像素的点: {(reproj_errors < 1).sum()} ({(reproj_errors < 1).mean()*100:.1f}%)")
    print(f"  误差 < 5 像素的点: {(reproj_errors < 5).sum()} ({(reproj_errors < 5).mean()*100:.1f}%)")
    print(f"  误差 < 10 像素的点: {(reproj_errors < 10).sum()} ({(reproj_errors < 10).mean()*100:.1f}%)")
    
    # 检查是否有异常大的误差
    large_error_mask = reproj_errors > 50
    if large_error_mask.any():
        print(f"\n  ⚠️  警告: {large_error_mask.sum()} 个点的重投影误差 > 50 像素")
        print(f"  这可能表明存在坐标系转换问题")
    
    # 额外诊断：检查是否有 Z 值异常小的点（可能导致归一化坐标异常大）
    z_small_mask = points_cam[2, :] < 0.01
    if z_small_mask.any():
        print(f"\n  ⚠️  警告: {z_small_mask.sum()} 个点的相机坐标系 Z 值 < 0.01")
        print(f"  这些点的归一化坐标会异常大，导致投影错误")
        print(f"  这些点的 Z 值范围: [{points_cam[2, z_small_mask].min():.6f}, {points_cam[2, z_small_mask].max():.6f}]")
    
    return {
        "reproj_errors": reproj_errors,
        "projected_points": projected_points,
        "world_points_at_tracks": pred_points_3d,
    }


def verify_projection_from_depth(depth_map, extrinsic, intrinsic, pred_tracks, image_shape):
    """从深度图验证投影正确性"""
    print("\n" + "="*60)
    print("从深度图验证投影")
    print("="*60)
    
    # 处理深度图的形状：可能是 (H, W) 或 (H, W, 1)
    if depth_map.ndim == 3 and depth_map.shape[2] == 1:
        # 如果是 (H, W, 1)，先squeeze
        depth_map = depth_map.squeeze(-1)
    elif depth_map.ndim > 2:
        # 如果是批次格式，取第一帧
        if depth_map.ndim == 4:
            depth_map = depth_map[0]
            if depth_map.shape[2] == 1:
                depth_map = depth_map.squeeze(-1)
        elif depth_map.ndim == 3 and depth_map.shape[0] == 1:
            depth_map = depth_map[0]
    
    H, W = depth_map.shape[:2]
    
    # 从深度图反投影到世界坐标（直接使用单帧函数）
    from vggt.utils.geometry import depth_to_world_coords_points
    world_points, _, _ = depth_to_world_coords_points(
        depth_map,   # [H, W]
        extrinsic,  # [3, 4]
        intrinsic   # [3, 3]
    )  # [H, W, 3]
    
    # 从轨迹点获取对应的 3D 点
    # 注意：轨迹点坐标是浮点数（特征匹配的亚像素精度），应该使用双线性插值而不是直接索引
    # 创建深度图的网格
    y_grid, x_grid = np.mgrid[0:H, 0:W]
    points_grid = np.column_stack([x_grid.ravel(), y_grid.ravel()])  # [H*W, 2]
    world_points_flat = world_points.reshape(-1, 3)  # [H*W, 3]
    
    # 使用双线性插值获取轨迹点位置的 3D 点
    try:
        from scipy.interpolate import griddata
        world_points_at_tracks = griddata(
            points_grid, world_points_flat,
            pred_tracks, method='linear', fill_value=np.nan
        )  # [N, 3]
        
        # 检查插值结果
        valid_mask = ~np.isnan(world_points_at_tracks).any(axis=1)
        if valid_mask.sum() < len(pred_tracks) * 0.9:
            print(f"  ⚠️  警告: {len(pred_tracks) - valid_mask.sum()} 个点插值失败（可能在图像边界外）")
            # 对于插值失败的点，使用最近邻插值作为后备
            world_points_at_tracks_fallback = griddata(
                points_grid, world_points_flat,
                pred_tracks[~valid_mask], method='nearest', fill_value=0
            )
            world_points_at_tracks[~valid_mask] = world_points_at_tracks_fallback
            valid_mask = ~np.isnan(world_points_at_tracks).any(axis=1)
    except ImportError:
        # 如果没有 scipy，使用直接索引（精度较低）
        print("  ⚠️  警告: 未安装 scipy，使用直接索引（精度较低）")
        track_indices = pred_tracks.astype(int)
        track_indices[:, 0] = np.clip(track_indices[:, 0], 0, W-1)
        track_indices[:, 1] = np.clip(track_indices[:, 1], 0, H-1)
        world_points_at_tracks = world_points[track_indices[:, 1], track_indices[:, 0]]  # [N, 3]
        valid_mask = np.ones(len(pred_tracks), dtype=bool)
    
    # 只使用有效的点进行投影
    world_points_at_tracks_valid = world_points_at_tracks[valid_mask]
    pred_tracks_valid = pred_tracks[valid_mask]
    
    # 将 3D 点投影回 2D
    projected_points_valid, _ = project_3D_points_np(
        world_points_at_tracks_valid,  # [N_valid, 3]
        extrinsic[None],               # [1, 3, 4]
        intrinsic[None]                 # [1, 3, 3]
    )
    projected_points_valid = projected_points_valid[0]  # [N_valid, 2]
    
    # 创建完整的投影点数组（无效点设为 NaN）
    projected_points = np.full((len(pred_tracks), 2), np.nan)
    projected_points[valid_mask] = projected_points_valid
    
    # 计算重投影误差（只计算有效点）
    reproj_errors = np.full(len(pred_tracks), np.nan)
    reproj_errors[valid_mask] = np.linalg.norm(projected_points_valid - pred_tracks_valid, axis=1)
    
    valid_errors = reproj_errors[valid_mask]
    print(f"轨迹点数量: {len(pred_tracks)} (有效点: {valid_mask.sum()})")
    print(f"重投影误差统计（仅有效点）:")
    print(f"  平均误差: {valid_errors.mean():.3f} 像素")
    print(f"  中位数误差: {np.median(valid_errors):.3f} 像素")
    print(f"  最大误差: {valid_errors.max():.3f} 像素")
    print(f"  最小误差: {valid_errors.min():.3f} 像素")
    print(f"  误差 < 1 像素的点: {(valid_errors < 1).sum()} ({(valid_errors < 1).sum()/len(valid_errors)*100:.1f}%)")
    print(f"  误差 < 5 像素的点: {(valid_errors < 5).sum()} ({(valid_errors < 5).sum()/len(valid_errors)*100:.1f}%)")
    print(f"  误差 < 10 像素的点: {(valid_errors < 10).sum()} ({(valid_errors < 10).sum()/len(valid_errors)*100:.1f}%)")
    
    # 检查是否有异常大的误差
    large_error_mask = valid_errors > 50
    if large_error_mask.any():
        print(f"\n  ⚠️  警告: {large_error_mask.sum()} 个有效点的重投影误差 > 50 像素")
        print(f"  这可能表明存在坐标系转换问题")
    
    return {
        "reproj_errors": reproj_errors,
        "projected_points": projected_points,
        "world_points_at_tracks": world_points_at_tracks,
    }


def visualize_projection_comparison(image_path, pred_tracks, projected_points, reproj_errors, output_path):
    """可视化投影结果对比"""
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"无法加载图像: {image_path}")
        return
    
    H, W = image.shape[:2]
    
    # 创建可视化图像
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 原始图像 + 轨迹点
    ax = axes[0]
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    valid_mask = (pred_tracks[:, 0] >= 0) & (pred_tracks[:, 0] < W) & \
                 (pred_tracks[:, 1] >= 0) & (pred_tracks[:, 1] < H)
    if valid_mask.any():
        ax.scatter(pred_tracks[valid_mask, 0], pred_tracks[valid_mask, 1], 
                  c='green', s=10, alpha=0.6, label='轨迹点 (pred_tracks)')
    ax.set_title('原始轨迹点')
    ax.axis('off')
    ax.legend()
    
    # 原始图像 + 投影点
    ax = axes[1]
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    valid_mask = (projected_points[:, 0] >= 0) & (projected_points[:, 0] < W) & \
                 (projected_points[:, 1] >= 0) & (projected_points[:, 1] < H)
    if valid_mask.any():
        ax.scatter(projected_points[valid_mask, 0], projected_points[valid_mask, 1], 
                  c='red', s=10, alpha=0.6, label='投影点 (projected)')
    ax.set_title('从 3D 投影的点')
    ax.axis('off')
    ax.legend()
    
    # 对比图：轨迹点 vs 投影点
    ax = axes[2]
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    valid_mask = (pred_tracks[:, 0] >= 0) & (pred_tracks[:, 0] < W) & \
                 (pred_tracks[:, 1] >= 0) & (pred_tracks[:, 1] < H) & \
                 (projected_points[:, 0] >= 0) & (projected_points[:, 0] < W) & \
                 (projected_points[:, 1] >= 0) & (projected_points[:, 1] < H)
    if valid_mask.any():
        ax.scatter(pred_tracks[valid_mask, 0], pred_tracks[valid_mask, 1], 
                  c='green', s=10, alpha=0.6, label='轨迹点')
        ax.scatter(projected_points[valid_mask, 0], projected_points[valid_mask, 1], 
                  c='red', s=10, alpha=0.6, label='投影点')
        # 绘制连接线
        for i in range(min(100, valid_mask.sum())):  # 只绘制前100个点
            idx = np.where(valid_mask)[0][i]
            ax.plot([pred_tracks[idx, 0], projected_points[idx, 0]], 
                   [pred_tracks[idx, 1], projected_points[idx, 1]], 
                   'b-', alpha=0.3, linewidth=0.5)
    ax.set_title('对比：轨迹点 (绿) vs 投影点 (红)')
    ax.axis('off')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n可视化结果已保存到: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="验证对应点投影的正确性")
    parser.add_argument("--vggt_dir", type=str, required=True,
                       help="VGGT 输出目录（包含 frame_XXX_tracks.npz 和 frame_XXX.json）")
    parser.add_argument("--images_dir", type=str, default=None,
                       help="图像目录（用于可视化，可选）")
    parser.add_argument("--frame_idx", type=int, default=0,
                       help="要验证的帧索引")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="输出目录（默认：vggt_dir/verification）")
    
    args = parser.parse_args()
    
    # 设置输出目录
    if args.output_dir is None:
        args.output_dir = os.path.join(args.vggt_dir, "verification")
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*60)
    print("对应点投影验证")
    print("="*60)
    print(f"VGGT 目录: {args.vggt_dir}")
    print(f"帧索引: {args.frame_idx}")
    print(f"输出目录: {args.output_dir}")
    
    # 加载数据
    print("\n加载 VGGT 数据...")
    data = load_vggt_data(args.vggt_dir, args.frame_idx)
    
    pred_tracks = data["pred_tracks"]
    pred_points_3d = data.get("pred_points_3d", None)  # 轨迹点对应的 3D 点
    extrinsic = data["extrinsic"]
    intrinsic = data["intrinsic"]
    depth_map = data["depth_map"]
    
    print(f"  轨迹点数量: {len(pred_tracks)}")
    print(f"  外参矩阵形状: {extrinsic.shape}")
    print(f"  内参矩阵形状: {intrinsic.shape}")
    if depth_map is not None:
        print(f"  深度图形状: {depth_map.shape}")
    
    # 验证外参矩阵
    ext_verify = verify_extrinsic_matrix(extrinsic)
    
    # 优先使用 pred_points_3d 进行验证（如果可用）
    proj_verify = None
    if pred_points_3d is not None:
        print("\n使用轨迹点对应的 3D 点进行验证（推荐）...")
        proj_verify = verify_projection_from_points_3d(
            pred_tracks, pred_points_3d, extrinsic, intrinsic
        )
        
        # 对比测试：使用当前帧深度图反投影的 3D 点
        if depth_map is not None:
            print("\n" + "="*60)
            print("对比测试：使用当前帧深度图反投影的 3D 点")
            print("="*60)
            depth_verify = verify_projection_from_depth(
                depth_map, extrinsic, intrinsic, pred_tracks, 
                image_shape=(depth_map.shape[0], depth_map.shape[1])
            )
            if depth_verify is not None and "reproj_errors" in depth_verify:
                depth_error = depth_verify["reproj_errors"].mean()
                pred_error = proj_verify["reproj_errors"].mean()
                print(f"\n对比结果:")
                print(f"  使用 pred_points_3d（来自查询帧）: 平均误差 {pred_error:.2f} 像素")
                print(f"  使用深度图反投影（当前帧）: 平均误差 {depth_error:.2f} 像素")
                if depth_error < pred_error * 0.5:
                    print(f"  ⚠️  深度图反投影误差更小，说明 pred_points_3d 可能来自错误的坐标系")
                    print(f"  可能原因: pred_points_3d 来自查询帧的坐标系，而当前帧使用不同的坐标系")
                elif pred_error < depth_error * 0.5:
                    print(f"  ✅ pred_points_3d 误差更小，说明它更准确")
                else:
                    print(f"  ⚠️  两种方法误差都很大，可能是投影计算或外参/内参有问题")
    elif depth_map is not None:
        print("\n使用深度图进行验证（可能不够准确）...")
        proj_verify = verify_projection_from_depth(
            depth_map, extrinsic, intrinsic, pred_tracks, 
            image_shape=(depth_map.shape[0], depth_map.shape[1])
        )
    else:
        print("\n⚠️  未找到 pred_points_3d 或深度图，无法进行投影验证")
    
    # 可视化（如果有图像和验证结果）
    if proj_verify is not None and args.images_dir is not None and "projected_points" in proj_verify:
        images_dir = Path(args.images_dir)
        image_files = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.JPG")) + \
                     sorted(images_dir.glob("*.png")) + sorted(images_dir.glob("*.PNG"))
        
        if args.frame_idx < len(image_files):
            image_path = image_files[args.frame_idx]
            output_path = os.path.join(args.output_dir, f"frame_{args.frame_idx:03d}_projection_verification.png")
            visualize_projection_comparison(
                image_path, pred_tracks, 
                proj_verify["projected_points"], 
                proj_verify["reproj_errors"],
                output_path
            )
    
    # 总结
    print("\n" + "="*60)
    print("验证总结")
    print("="*60)
    
    if ext_verify["is_orthogonal"]:
        print("✅ 外参矩阵旋转部分有效（正交矩阵）")
    else:
        print("❌ 外参矩阵旋转部分无效（不是正交矩阵）")
    
    if abs(ext_verify["det"] - 1.0) < 0.01:
        print("✅ 旋转矩阵行列式正常（接近 1.0）")
    else:
        print("❌ 旋转矩阵行列式异常")
    
    if proj_verify is not None and "reproj_errors" in proj_verify:
        mean_error = proj_verify["reproj_errors"].mean()
        if mean_error < 5.0:
            print(f"✅ 重投影误差较小（平均 {mean_error:.2f} 像素）")
        elif mean_error < 20.0:
            print(f"⚠️  重投影误差中等（平均 {mean_error:.2f} 像素），可能存在轻微问题")
        else:
            print(f"❌ 重投影误差较大（平均 {mean_error:.2f} 像素），可能存在坐标系转换问题")
    
    print(f"\n详细结果已保存到: {args.output_dir}")


if __name__ == "__main__":
    main()

