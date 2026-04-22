# 3DGS-VGGT: VGGT 几何可靠性权重指导的 3D Gaussian Splatting

基于 VGGT 几何可靠性权重图（W_geo）改进 3D Gaussian Splatting 训练流程。

## 核心创新

将原有的单一权重图 W 拆分为两个语义独立的权重图：

- **W_geo**（几何可靠性）→ 加权深度正则化损失
- **W_appear**（外观可建模性）→ 加权光度损失（已实现，待验证）

公式：`W_geo = 0.5·(1-σ) + 0.3·V + 0.2·C`

## 实验结果

在 Kitchen、Fern、Flower 三个场景上验证 W_geo 方案（7000 iter）：

| 场景 | Baseline PSNR | W_geo PSNR | 提升 |
|------|---------------|------------|------|
| Kitchen | 40.34 | 40.40 | +0.06 dB |
| Fern | 29.13 | 31.05 | **+1.92 dB** |
| Flower | 21.25 | 21.75 | +0.50 dB |

## 使用方法

```bash
# 1. VGGT 推理生成深度/权重图
python vggt/export_vggt_for_3dgs.py --scene_dir vggt/examples/<scene>

# 2. 转换为 3DGS 格式
python vggt/convert_vggt_to_3dgs.py --vggt_dir vggt/examples/<scene>/vggt --output_dir vggt/examples/<scene>/3dgs_input

# 3. 训练 3DGS
python gaussian-splatting-main/train.py \
  -s vggt/examples/<scene>/3dgs_input \
  -d depths --weights weights \
  -m output/<scene> \
  --iterations 7000
```

## 目录结构

- `vggt/` — VGGT 代码与集成脚本
- `gaussian-splatting-main/` — 3DGS 代码（已加入权重图支持）

## 致谢

- [VGGT](https://github.com/facebookresearch/vggt) - Meta
- [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) - Inria
