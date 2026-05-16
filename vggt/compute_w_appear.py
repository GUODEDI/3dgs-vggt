"""
计算 W_appear: 多视角光度一致性权重图

原理：
  对每个像素，通过深度反投影到3D，再投影到其他帧，
  比较各帧中该位置的颜色。颜色方差大 = 视角依赖外观 = 3DGS难建模 = W_appear低
"""

import numpy as np
import cv2
import json
import os
from pathlib import Path


def load_frame_data(vggt_dir, images_dir, num_frames):
    """加载所有帧的图像、深度、相机参数"""
    frames = []
    for i in range(num_frames):
        json_path = os.path.join(vggt_dir, f"frame_{i:03d}.json")
        depth_path = os.path.join(vggt_dir, f"frame_{i:03d}_depth.npy")

        if not os.path.exists(json_path) or not os.path.exists(depth_path):
            continue

        with open(json_path, 'r') as f:
            meta = json.load(f)

        depth = np.load(depth_path)
        H, W = depth.shape[:2]

        # 加载图像
        img_name = os.path.basename(meta["original_image_path"])
        img_path = os.path.join(images_dir, img_name)
        img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (W, H)).astype(np.float32) / 255.0
        else:
            img = np.ones((H, W, 3), dtype=np.float32) * 0.5

        intr = meta["intrinsic"]
        extr = meta["extrinsic"]
        R = np.array(extr["R"])
        t = np.array(extr["t"])

        K = np.array([
            [intr["fx"], 0, intr["cx"]],
            [0, intr["fy"], intr["cy"]],
            [0, 0, 1]
        ])

        frames.append({
            "image": img,
            "depth": depth,
            "R_w2c": R,
            "t_w2c": t,
            "K": K,
            "H": H,
            "W": W,
            "name": img_name
        })

    return frames


def compute_w_appear_for_frame(ref_idx, frames, step=4):
    """
    计算单帧的 W_appear

    对参考帧的每个像素:
    1. 用深度反投影到3D世界坐标
    2. 投影到其他帧
    3. 采样颜色
    4. 计算颜色方差
    """
    ref = frames[ref_idx]
    H, W = ref["H"], ref["W"]
    depth = ref["depth"]
    K_ref = ref["K"]
    R_ref = ref["R_w2c"]
    t_ref = ref["t_w2c"]

    # 相机到世界变换
    R_c2w = R_ref.T
    t_c2w = -R_ref.T @ t_ref

    # 生成像素网格 (降采样计算，再上采样)
    h_down = H // step
    w_down = W // step
    vs = np.linspace(0, H - 1, h_down).astype(np.float32)
    us = np.linspace(0, W - 1, w_down).astype(np.float32)
    uu, vv = np.meshgrid(us, vs)  # [h_down, w_down]

    # 在原图上采样深度
    u_int = np.clip(uu.astype(int), 0, W - 1)
    v_int = np.clip(vv.astype(int), 0, H - 1)
    d = depth[v_int, u_int]  # [h_down, w_down]

    # 像素 -> 相机坐标
    fx, fy = K_ref[0, 0], K_ref[1, 1]
    cx, cy = K_ref[0, 2], K_ref[1, 2]
    x_cam = (uu - cx) * d / fx
    y_cam = (vv - cy) * d / fy
    z_cam = d

    # 相机 -> 世界坐标
    pts_cam = np.stack([x_cam, y_cam, z_cam], axis=-1)  # [h, w, 3]
    pts_world = np.einsum('ij,hwj->hwi', R_c2w, pts_cam) + t_c2w[None, None, :]

    # 参考帧自己的颜色
    ref_colors = ref["image"][v_int, u_int]  # [h, w, 3]

    # 收集所有帧的颜色
    all_colors = [ref_colors]
    valid_count = np.ones((h_down, w_down), dtype=np.float32)

    for j, frame in enumerate(frames):
        if j == ref_idx:
            continue

        K_j = frame["K"]
        R_j = frame["R_w2c"]
        t_j = frame["t_w2c"]
        H_j, W_j = frame["H"], frame["W"]

        # 世界 -> 目标相机坐标
        pts_cam_j = np.einsum('ij,hwj->hwi', R_j, pts_world) + t_j[None, None, :]

        # 深度检查(在相机前方)
        z_j = pts_cam_j[:, :, 2]

        # 目标相机 -> 像素
        u_j = K_j[0, 0] * pts_cam_j[:, :, 0] / z_j + K_j[0, 2]
        v_j = K_j[1, 1] * pts_cam_j[:, :, 1] / z_j + K_j[1, 2]

        # 基本有效性检查（在画面内、在相机前方）
        valid = (z_j > 0.01) & \
                (u_j >= 0) & (u_j < W_j - 1) & \
                (v_j >= 0) & (v_j < H_j - 1)

        # 采样坐标
        u_j_int = np.clip(u_j.astype(int), 0, W_j - 1)
        v_j_int = np.clip(v_j.astype(int), 0, H_j - 1)

        # 深度一致性校验：目标帧该位置的实际深度 vs 投影期望深度
        depth_j = frame["depth"]
        d_actual = depth_j[v_j_int, u_j_int]  # 目标帧该位置的实际深度
        d_expected = z_j  # 投影过去的期望深度
        # 相对深度误差 < 10% 才认为投影可靠
        depth_consistent = np.abs(d_actual - d_expected) / (np.abs(d_expected) + 1e-6) < 0.1
        valid = valid & depth_consistent

        # 采样颜色(最近邻)
        colors_j = frame["image"][v_j_int, u_j_int]  # [h, w, 3]

        # 无效位置用参考帧颜色(不影响方差)
        colors_j[~valid] = ref_colors[~valid]
        valid_count += valid.astype(np.float32)

        all_colors.append(colors_j)

    # 计算颜色方差 [h, w, N, 3]
    color_stack = np.stack(all_colors, axis=2)  # [h, w, N, 3]
    color_var = np.var(color_stack, axis=2).mean(axis=-1)  # [h, w] RGB方差取平均

    # Bug 4 修复：区分"无有效投影"和"颜色一致"两种情况
    # 无有效投影（valid_count < 2）:无法判断,标记为 invalid_mask,后续设为 W_appear = 1.0
    # 颜色一致(色彩方差为 0):说明真正朗伯面,正常归一化为 W_appear ≈ 1.0
    invalid_mask = valid_count < 2

    # 归一化方差到 0-1 (用 percentile 避免离群值)
    # 仅基于有效像素的非零方差计算 p95,避免无效像素的 0 干扰统计
    valid_var = color_var[(~invalid_mask) & (color_var > 0)]
    if valid_var.size > 0:
        p95 = np.percentile(valid_var, 95)
    else:
        # 全场景均为朗伯面/无效:归一化系数取 1.0,但保证不会让所有像素一致
        p95 = 1.0
    p95 = max(p95, 1e-6)
    var_normalized = np.clip(color_var / p95, 0, 1)

    # W_appear = 1 - normalized_variance
    # 方差大 = 视角依赖 = 3DGS 难 = W_appear 低
    w_appear_down = 1.0 - var_normalized

    # 无效投影区域: 不参与计算, 保持 W_appear = 1.0(不降权)
    w_appear_down[invalid_mask] = 1.0

    # 上采样到原始分辨率
    w_appear = cv2.resize(w_appear_down.astype(np.float32), (W, H),
                          interpolation=cv2.INTER_LINEAR)
    w_appear = np.clip(w_appear, 0.1, 1.0)

    return w_appear


def generate_w_appear(vggt_dir, images_dir, output_dir, step=4):
    """生成所有帧的 W_appear"""
    # 统计帧数
    frame_files = sorted(Path(vggt_dir).glob("frame_*.json"))
    num_frames = len(frame_files)
    print(f"Loading {num_frames} frames...")

    frames = load_frame_data(vggt_dir, images_dir, num_frames)
    print(f"Loaded {len(frames)} frames")

    os.makedirs(output_dir, exist_ok=True)

    for i in range(len(frames)):
        w_appear = compute_w_appear_for_frame(i, frames, step=step)

        # 保存
        img_stem = os.path.splitext(frames[i]["name"])[0]
        out_path = os.path.join(output_dir, f"{img_stem}.npy")
        np.save(out_path, w_appear.astype(np.float32))

        print(f"  frame {i+1}: mean={w_appear.mean():.3f}, "
              f"min={w_appear.min():.3f}, max={w_appear.max():.3f}, "
              f"<0.5={((w_appear < 0.5).sum() / w_appear.size * 100):.1f}%")

    print("Done.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--vggt_dir", type=str, required=True)
    parser.add_argument("--images_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--step", type=int, default=4, help="Downsample step for computation")
    args = parser.parse_args()
    generate_w_appear(args.vggt_dir, args.images_dir, args.output_dir, args.step)
