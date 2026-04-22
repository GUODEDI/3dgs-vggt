#!/usr/bin/env python3
"""
将 VGGT 导出的数据转换为 3DGS 可用的格式

功能：
1. 将 VGGT 的 JSON 相机参数转换为 COLMAP 格式
2. 将深度图从 NPY 转换为 PNG 格式
3. 生成 depth_params.json 文件
4. 创建 3DGS 需要的目录结构
"""

import os
import json
import numpy as np
import cv2
from pathlib import Path
from scipy.spatial.transform import Rotation
import struct
from collections import namedtuple

# COLMAP 数据结构
CameraModel = namedtuple("CameraModel", ["model_id", "model_name", "num_params"])
Camera = namedtuple("Camera", ["id", "model", "width", "height", "params"])
Image = namedtuple("Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])

CAMERA_MODELS = {
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
}
CAMERA_MODEL_IDS = {camera_model.model_id: camera_model for camera_model in CAMERA_MODELS}
CAMERA_MODEL_NAMES = {camera_model.model_name: camera_model for camera_model in CAMERA_MODELS}


def rotmat2qvec(R):
    """将旋转矩阵转换为四元数（COLMAP 格式）"""
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


def write_cameras_binary(cameras, path):
    """写入 COLMAP cameras.bin 文件"""
    with open(path, "wb") as fid:
        fid.write(struct.pack("<Q", len(cameras)))
        for camera in cameras:
            fid.write(struct.pack("<IIQQ", camera.id, camera.model.model_id, camera.width, camera.height))
            fid.write(struct.pack(f"<{len(camera.params)}d", *camera.params))


def write_images_binary(images, path):
    """写入 COLMAP images.bin 文件"""
    with open(path, "wb") as fid:
        fid.write(struct.pack("<Q", len(images)))
        for image in images:
            # 格式: i (image_id) + dddd (qvec) + ddd (tvec) + i (camera_id) = idddddddi (9个元素)
            fid.write(struct.pack("<idddddddi", image.id, *image.qvec, *image.tvec, image.camera_id))
            # 写入图像名称（以 \x00 结尾）
            fid.write(image.name.encode("utf-8"))
            fid.write(b"\x00")
            # 写入 2D 点数量
            num_points2D = len(image.xys)
            fid.write(struct.pack("<Q", num_points2D))
            # 写入 2D 点坐标和对应的 3D 点 ID (格式: ddq 对于每个点)
            if num_points2D > 0:
                for xy, p3d_id in zip(image.xys, image.point3D_ids):
                    fid.write(struct.pack("<ddq", float(xy[0]), float(xy[1]), int(p3d_id)))


def write_points3D_binary(points3D, path):
    """写入 COLMAP points3D.bin 文件（空点云，3DGS 会自己生成）"""
    with open(path, "wb") as fid:
        fid.write(struct.pack("<Q", len(points3D)))
        for point in points3D:
            fid.write(struct.pack("<QdddBBBd", point.id, *point.xyz, *point.rgb, point.error))
            fid.write(struct.pack("<Q", len(point.image_ids)))
            if len(point.image_ids) > 0:
                fid.write(point.image_ids.tobytes())
                fid.write(point.point2D_idxs.tobytes())


def convert_depth_npy_to_png(depth_npy_path, depth_png_path, scale_factor=65535.0):
    """将深度图从 NPY 转换为 PNG 格式（16位）"""
    depth = np.load(depth_npy_path)
    
    # 归一化深度值到 0-1 范围
    depth_min = depth[depth > 0].min() if (depth > 0).any() else depth.min()
    depth_max = depth.max()
    
    if depth_max > depth_min:
        depth_norm = (depth - depth_min) / (depth_max - depth_min)
    else:
        depth_norm = np.zeros_like(depth)
    
    # 转换为 16 位整数
    depth_16bit = (depth_norm * scale_factor).astype(np.uint16)
    
    # 保存为 PNG (use imencode to handle non-ASCII paths)
    success, buf = cv2.imencode('.png', depth_16bit)
    if success:
        buf.tofile(str(depth_png_path))
    else:
        raise IOError(f"Failed to encode depth image: {depth_png_path}")
    
    return depth_min, depth_max


def convert_vggt_to_colmap(vggt_dir, output_dir, scene_name="scene"):
    """
    将 VGGT 导出的数据转换为 COLMAP 格式
    
    Args:
        vggt_dir: VGGT 输出目录（包含 frame_XXX.json 和 frame_XXX_depth.npy）
        output_dir: 输出目录（3DGS 场景目录）
        scene_name: 场景名称
    """
    vggt_dir = Path(vggt_dir)
    output_dir = Path(output_dir)
    
    # 创建输出目录结构
    sparse_dir = output_dir / "sparse" / "0"
    sparse_dir.mkdir(parents=True, exist_ok=True)
    
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    depths_dir = output_dir / "depths"
    depths_dir.mkdir(parents=True, exist_ok=True)
    
    weights_dir = output_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    
    # 读取所有帧的 JSON 文件
    frame_files = sorted(vggt_dir.glob("frame_*.json"))
    
    if len(frame_files) == 0:
        raise ValueError(f"在 {vggt_dir} 中未找到 frame_*.json 文件")
    
    print(f"找到 {len(frame_files)} 帧数据")
    
    cameras = []
    images = []
    depth_params = {}
    
    # 处理每一帧
    for frame_idx, frame_json_path in enumerate(frame_files):
        with open(frame_json_path, 'r') as f:
            frame_data = json.load(f)
        
        # 提取相机参数
        intrinsic = frame_data["intrinsic"]
        extrinsic = frame_data["extrinsic"]
        image_size = frame_data["image_size"]
        original_image_path = frame_data["original_image_path"]
        
        # 创建相机（所有帧共享同一个相机）
        if frame_idx == 0:
            camera_id = 1
            camera = Camera(
                id=camera_id,
                model=CAMERA_MODEL_NAMES["PINHOLE"],
                width=image_size[0],
                height=image_size[1],
                params=[intrinsic["fx"], intrinsic["fy"], intrinsic["cx"], intrinsic["cy"]]
            )
            cameras.append(camera)
        
        # 转换外参（从 OpenCV 格式转换为 COLMAP 格式）
        # VGGT 的 extrinsic 是 [R|t]，表示从世界到相机的变换（OpenCV 约定）
        R = np.array(extrinsic["R"])
        t = np.array(extrinsic["t"])
        
        # COLMAP 也使用世界到相机的变换，但格式是四元数 + 平移向量
        # 直接使用 R 和 t（已经是世界到相机的变换）
        R_w2c = R
        t_w2c = t
        
        # 转换为四元数
        qvec = rotmat2qvec(R_w2c)
        
        # 获取图像名称
        image_name = os.path.basename(original_image_path)
        
        # 创建 Image 对象
        image = Image(
            id=frame_idx + 1,  # COLMAP 使用 1-based 索引
            qvec=qvec,
            tvec=t_w2c,
            camera_id=camera_id,
            name=image_name,
            xys=np.array([], dtype=np.float32).reshape(0, 2),  # 空数组
            point3D_ids=np.array([], dtype=np.int64)  # 空数组
        )
        images.append(image)
        
        # 复制原始图像到 images 目录
        original_image_full_path = vggt_dir.parent / original_image_path
        if not original_image_full_path.exists():
            # 尝试在 vggt_dir 的父目录查找
            original_image_full_path = vggt_dir.parent.parent / original_image_path
        if original_image_full_path.exists():
            import shutil
            dest_image_path = images_dir / image_name
            shutil.copy2(original_image_full_path, dest_image_path)
            print(f"  复制图像: {image_name}")
        else:
            print(f"  [WARN] Image not found: {original_image_path}")
        
        # 转换深度图
        depth_npy_path = vggt_dir / f"frame_{frame_idx:03d}_depth.npy"
        if depth_npy_path.exists():
            depth_png_path = depths_dir / f"{image_name.replace('.png', '').replace('.jpg', '').replace('.jpeg', '')}.png"
            depth_min, depth_max = convert_depth_npy_to_png(depth_npy_path, depth_png_path)
            
            # 计算深度缩放参数（简化版本，使用中位数缩放）
            depth = np.load(depth_npy_path)
            depth_valid = depth[depth > 0]
            if len(depth_valid) > 0:
                median_depth = np.median(depth_valid)
                # 简化的缩放：假设深度图已经归一化，使用中位数作为参考
                scale = 1.0 / median_depth if median_depth > 0 else 1.0
                offset = 0.0
            else:
                scale = 1.0
                offset = 0.0
            
            # 3DGS uses filename without extension as key
            image_stem = os.path.splitext(image_name)[0]
            depth_params[image_stem] = {
                "scale": float(scale),
                "offset": float(offset)
            }
        
        # 转换权重图（保存为 NPY 格式，供 3DGS 使用）
        weight_npy_path = vggt_dir / f"frame_{frame_idx:03d}_W.npy"
        if weight_npy_path.exists():
            weight = np.load(weight_npy_path)
            # 保存权重图（保持原始分辨率，NPY 格式）
            weight_output_path = weights_dir / f"{image_name.replace('.png', '.npy').replace('.jpg', '.npy').replace('.jpeg', '.npy')}"
            np.save(weight_output_path, weight)
            print(f"  保存权重图: {weight_output_path.name}")
    
    # 计算中位数缩放（用于所有深度图）
    if depth_params:
        all_scales = np.array([p["scale"] for p in depth_params.values() if p["scale"] > 0])
        if len(all_scales) > 0:
            med_scale = np.median(all_scales)
            for key in depth_params:
                depth_params[key]["med_scale"] = float(med_scale)
    
    # 写入 COLMAP 文件
    print("写入 COLMAP 文件...")
    write_cameras_binary(cameras, sparse_dir / "cameras.bin")
    write_images_binary(images, sparse_dir / "images.bin")
    
    # 从深度图反投影生成初始3D点云
    Point3D = namedtuple("Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])
    all_points_3d = []
    all_points_rgb = []

    for frame_idx, frame_json_path in enumerate(frame_files):
        with open(frame_json_path, 'r') as f:
            frame_data = json.load(f)

        intrinsic = frame_data["intrinsic"]
        extrinsic = frame_data["extrinsic"]
        R = np.array(extrinsic["R"])
        t = np.array(extrinsic["t"])

        depth_npy_path = vggt_dir / f"frame_{frame_idx:03d}_depth.npy"
        if not depth_npy_path.exists():
            continue
        depth = np.load(depth_npy_path)
        H, W = depth.shape[:2]

        # 读取对应图像获取颜色
        image_name = os.path.basename(frame_data["original_image_path"])
        img_path = images_dir / image_name
        if img_path.exists():
            img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (W, H))
            else:
                img = np.ones((H, W, 3), dtype=np.uint8) * 128
        else:
            img = np.ones((H, W, 3), dtype=np.uint8) * 128

        # 每帧均匀采样点（步长越大越稀疏）
        step = max(H, W) // 50  # 每帧约 50x50 = 2500 点
        step = max(step, 1)

        fx, fy = intrinsic["fx"], intrinsic["fy"]
        cx, cy = intrinsic["cx"], intrinsic["cy"]

        # 相机到世界的变换: R_c2w = R_w2c^T, t_c2w = -R_w2c^T @ t_w2c
        R_c2w = R.T
        t_c2w = -R.T @ t

        for v in range(0, H, step):
            for u in range(0, W, step):
                d = depth[v, u]
                if d <= 0 or np.isnan(d) or np.isinf(d):
                    continue
                # 像素坐标 -> 相机坐标
                x_cam = (u - cx) * d / fx
                y_cam = (v - cy) * d / fy
                z_cam = d
                pt_cam = np.array([x_cam, y_cam, z_cam])
                # 相机坐标 -> 世界坐标
                pt_world = R_c2w @ pt_cam + t_c2w
                all_points_3d.append(pt_world)
                all_points_rgb.append(img[v, u])

    # 过滤异常点
    if len(all_points_3d) > 0:
        pts = np.array(all_points_3d)
        rgbs = np.array(all_points_rgb)
        # 移除距中心过远的离群点
        center = np.median(pts, axis=0)
        dists = np.linalg.norm(pts - center, axis=1)
        threshold = np.percentile(dists, 95) * 2
        valid = dists < threshold
        pts = pts[valid]
        rgbs = rgbs[valid]

        points3D_list = []
        for i in range(len(pts)):
            points3D_list.append(Point3D(
                id=i + 1,
                xyz=pts[i],
                rgb=rgbs[i].astype(np.uint8),
                error=0.0,
                image_ids=np.array([], dtype=np.int32),
                point2D_idxs=np.array([], dtype=np.int32)
            ))
        print(f"[OK] Generated {len(points3D_list)} initial 3D points from depth maps")
    else:
        points3D_list = []
        print("[WARN] No initial points generated, using empty point cloud")

    write_points3D_binary(points3D_list, sparse_dir / "points3D.bin")
    
    # 写入 depth_params.json
    if depth_params:
        with open(sparse_dir / "depth_params.json", 'w') as f:
            json.dump(depth_params, f, indent=2)
        print(f"[OK] Depth params saved: {sparse_dir / 'depth_params.json'}")

    print(f"\n[OK] Conversion done!")
    print(f"  输出目录: {output_dir}")
    print(f"  COLMAP 文件: {sparse_dir}")
    print(f"  深度图目录: {depths_dir}")
    print(f"  权重图目录: {weights_dir}")
    print(f"\n现在可以使用以下命令训练 3DGS：")
    print(f"  python train.py -s {output_dir} -d {depths_dir}")
    print(f"\n注意：权重图已保存在 {weights_dir}，但标准 3DGS 不支持权重图。")
    print(f"      请参考 'vggt/在3DGS中使用权重图W.md' 了解如何集成权重图。")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="将 VGGT 导出的数据转换为 3DGS 可用的 COLMAP 格式")
    parser.add_argument("--vggt_dir", type=str, required=True, 
                       help="VGGT 输出目录（包含 frame_XXX.json 和 frame_XXX_depth.npy）")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="3DGS 场景输出目录")
    parser.add_argument("--scene_name", type=str, default="scene",
                       help="场景名称")
    
    args = parser.parse_args()
    
    convert_vggt_to_colmap(args.vggt_dir, args.output_dir, args.scene_name)
