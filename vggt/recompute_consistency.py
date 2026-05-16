"""
仅重算 _cons.npy（多视角几何一致性数据），不重跑 VGGT 推理。
用于补全早期 VGGT 推理输出中缺失的 _cons.npy 文件。

输入：现有 vggt/ 目录中的：
  - frame_XXX_depth.npy（深度图）
  - frame_XXX_tracks.npz（轨迹点）
  - frame_XXX.json（相机参数 extrinsic + intrinsic）

输出：每帧的 frame_XXX_cons.npy
"""

import argparse
import json
import os
import sys
from pathlib import Path
import numpy as np


def unproject_depth_to_world(depth, K, R_w2c, t_w2c):
    """单帧反投影：深度 + 相机内参/外参 → 世界坐标 [H, W, 3]"""
    H, W = depth.shape[:2]
    if depth.ndim == 3 and depth.shape[2] == 1:
        depth = depth.squeeze(-1)

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    u = np.arange(W).reshape(1, W).astype(np.float32)
    v = np.arange(H).reshape(H, 1).astype(np.float32)

    x_cam = (u - cx) * depth / fx
    y_cam = (v - cy) * depth / fy
    z_cam = depth

    pts_cam = np.stack([x_cam, y_cam, z_cam], axis=-1)  # [H, W, 3]

    # 相机 → 世界（R_w2c 是世界到相机，所以世界到点 = R_w2c^T @ (cam - t)）
    R_c2w = R_w2c.T
    t_c2w = -R_w2c.T @ t_w2c
    pts_world = np.einsum('ij,hwj->hwi', R_c2w, pts_cam) + t_c2w[None, None, :]
    return pts_world


def project_world_to_image(world_points, K, R_w2c, t_w2c):
    """世界坐标 [N, 3] → 图像坐标 [N, 2]"""
    pts_cam = (R_w2c @ world_points.T + t_w2c[:, None]).T  # [N, 3]
    z = pts_cam[:, 2]
    valid = z > 1e-6
    u = K[0, 0] * pts_cam[:, 0] / np.where(valid, z, 1.0) + K[0, 2]
    v = K[1, 1] * pts_cam[:, 1] / np.where(valid, z, 1.0) + K[1, 2]
    return np.stack([u, v], axis=-1), valid


def compute_consistency_simple(world_points_per_frame, pred_tracks, extrinsics, intrinsics):
    """
    简化版 consistency 计算（基于 export_vggt_for_3dgs.py 的逻辑）
    每个轨迹点：
      用参考帧 ref 的 3D 位置 → 投影到其他帧 → 与轨迹点位置比较 → 误差越小一致性越高

    Args:
        world_points_per_frame: [S, H, W, 3] 每帧的世界坐标点
        pred_tracks: [S, N, 2] 轨迹点 2D 位置
        extrinsics: [S, 3, 4] 外参
        intrinsics: [S, 3, 3] 内参

    Returns:
        cons_per_track: [S, N] 每帧每轨迹点的一致性分数（0-1）
    """
    S, N = pred_tracks.shape[:2]
    H, W = world_points_per_frame.shape[1:3]

    cons_per_track = np.ones((S, N), dtype=np.float32)

    # Bug 7 修复：S<2 时无法做多视角一致性计算，返回全 1
    if S < 2:
        return cons_per_track

    ref_frame = 0

    # 用参考帧的世界坐标
    ref_world = world_points_per_frame[ref_frame]  # [H, W, 3]

    # 取参考帧轨迹点对应的 3D 位置
    ref_tracks = pred_tracks[ref_frame]  # [N, 2]
    ref_u = np.clip(ref_tracks[:, 0].astype(int), 0, W - 1)
    ref_v = np.clip(ref_tracks[:, 1].astype(int), 0, H - 1)
    ref_3d = ref_world[ref_v, ref_u]  # [N, 3]

    # Bug 6 修复：与 export_vggt_for_3dgs.py 中 compute_consistency_from_tracks 保持一致
    # 原公式：consistency[s] = 1 / (reproj_error + eps)，再做 per-frame min-max 归一化
    eps = 1e-8

    # 先按原公式计算 raw consistency
    for s in range(1, S):
        if s == ref_frame:
            continue
        proj_2d, valid = project_world_to_image(
            ref_3d, intrinsics[s], extrinsics[s, :3, :3], extrinsics[s, :3, 3]
        )
        actual_2d = pred_tracks[s]
        err = np.linalg.norm(proj_2d - actual_2d, axis=1)  # [N]
        # 与 export 一致：1 / (err + eps)
        raw_cons = 1.0 / (err + eps)
        # 无效投影位置使用大误差对应的分数（接近 0）
        raw_cons[~valid] = 0.0
        cons_per_track[s] = raw_cons

    # 与 export 一致：per-frame 最小-最大归一化到 [0, 1]
    for s in range(S):
        c_min, c_max = cons_per_track[s].min(), cons_per_track[s].max()
        if c_max > c_min:
            cons_per_track[s] = (cons_per_track[s] - c_min) / (c_max - c_min + 1e-8)

    return cons_per_track


def compute_visibility_from_tracks(pred_vis_scores, pred_tracks, image_shape):
    """从稀疏轨迹点插值到全图（双线性 / 高斯加权）"""
    from scipy.interpolate import griddata

    S, N = pred_tracks.shape[:2]
    H, W = image_shape

    full = np.zeros((S, H, W), dtype=np.float32)
    yy, xx = np.mgrid[0:H, 0:W]
    grid_pts = np.stack([xx.ravel(), yy.ravel()], axis=-1)  # [H*W, 2]

    for s in range(S):
        track_pts = pred_tracks[s]  # [N, 2]
        scores = pred_vis_scores[s]  # [N]

        # 过滤 NaN 或界外点
        valid = (track_pts[:, 0] >= 0) & (track_pts[:, 0] < W) & \
                (track_pts[:, 1] >= 0) & (track_pts[:, 1] < H) & \
                np.isfinite(track_pts).all(axis=1)
        if valid.sum() < 4:
            full[s] = np.ones((H, W), dtype=np.float32)
            continue

        try:
            interp = griddata(
                track_pts[valid], scores[valid], grid_pts,
                method='linear', fill_value=scores[valid].mean()
            ).reshape(H, W)
            full[s] = np.nan_to_num(interp, nan=scores[valid].mean()).astype(np.float32)
        except Exception:
            full[s] = np.ones((H, W), dtype=np.float32)

    return full


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vggt_dir", type=str, required=True)
    args = parser.parse_args()

    vggt_dir = Path(args.vggt_dir)
    if not vggt_dir.exists():
        raise FileNotFoundError(f"vggt_dir 不存在: {vggt_dir}")

    # 加载所有帧
    json_files = sorted(vggt_dir.glob("frame_*.json"))
    if not json_files:
        raise ValueError(f"无 frame_*.json 文件")

    S = len(json_files)
    print(f"找到 {S} 帧，开始重算 consistency...")

    # 加载相机参数
    intrinsics = []
    extrinsics = []
    image_shape = None
    depths = []
    for jf in json_files:
        with open(jf, 'r') as f:
            data = json.load(f)
        intr = data['intrinsic']
        K = np.array([[intr['fx'], 0, intr['cx']],
                      [0, intr['fy'], intr['cy']],
                      [0, 0, 1]], dtype=np.float64)
        extr = data['extrinsic']
        R = np.array(extr['R'], dtype=np.float64)
        t = np.array(extr['t'], dtype=np.float64).reshape(3)
        # 拼成 [3, 4]
        Rt = np.zeros((3, 4))
        Rt[:, :3] = R
        Rt[:, 3] = t
        intrinsics.append(K)
        extrinsics.append(Rt)

        # 加载 depth
        stem = jf.stem
        depth = np.load(vggt_dir / f"{stem}_depth.npy")
        if depth.ndim == 3:
            depth = depth.squeeze(-1)
        depths.append(depth)
        if image_shape is None:
            image_shape = depth.shape[:2]

    intrinsics = np.array(intrinsics)
    extrinsics = np.array(extrinsics)
    depths = np.array(depths)

    print(f"图像尺寸: {image_shape}, 帧数: {S}")

    # 加载所有帧的 tracks
    all_tracks = []
    all_vis = []
    for jf in json_files:
        stem = jf.stem
        npz = np.load(vggt_dir / f"{stem}_tracks.npz")
        all_tracks.append(npz['tracks'])
        all_vis.append(npz['vis_scores'])
    pred_tracks = np.stack(all_tracks, axis=0)  # [S, N, 2]
    pred_vis = np.stack(all_vis, axis=0)  # [S, N]
    print(f"轨迹: {pred_tracks.shape}")

    # 计算每帧世界坐标
    print("计算每帧 world_points...")
    world_points_all = []
    for s in range(S):
        wp = unproject_depth_to_world(depths[s], intrinsics[s], extrinsics[s, :3, :3], extrinsics[s, :3, 3])
        world_points_all.append(wp)
    world_points_all = np.stack(world_points_all, axis=0)  # [S, H, W, 3]

    # 计算每轨迹点的一致性
    print("计算 per-track consistency...")
    cons_per_track = compute_consistency_simple(world_points_all, pred_tracks, extrinsics, intrinsics)

    # 插值到全图
    print("插值 consistency 到全图...")
    cons_full = compute_visibility_from_tracks(cons_per_track, pred_tracks, image_shape)
    print(f"cons_full: {cons_full.shape}")

    # 保存
    for s in range(S):
        stem = json_files[s].stem
        out_path = vggt_dir / f"{stem}_cons.npy"
        np.save(out_path, cons_full[s].astype(np.float32))
    print(f"[OK] 已保存 {S} 个 _cons.npy 文件到 {vggt_dir}")


if __name__ == "__main__":
    main()
