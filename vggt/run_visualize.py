#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
运行轨迹可视化的辅助脚本（避免路径编码问题）
"""
import os
import sys
import subprocess

# 获取脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# 运行可视化脚本
cmd = [
    sys.executable,
    "visualize_tracks.py",
    "--tracks_dir", "examples/kitchen/vggt",
    "--images_dir", "examples/kitchen/images",
    "--color_by", "vis",
    "--point_size", "4",
    "--frame_idx", "0"
]

print("运行轨迹可视化...")
print(f"工作目录: {os.getcwd()}")
print(f"命令: {' '.join(cmd)}")
print()

result = subprocess.run(cmd, cwd=script_dir)

if result.returncode == 0:
    print("\n可视化完成！")
    print(f"结果保存在: {os.path.join(script_dir, 'examples/kitchen/vggt/track_visualizations')}")
else:
    print(f"\n错误: 退出码 {result.returncode}")




