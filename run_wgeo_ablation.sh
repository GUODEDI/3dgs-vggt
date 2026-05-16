#!/bin/bash
# W_geo 系数消融实验脚本
# 在 fern 场景上跑 5 组不同系数，比较最优配置
# 使用：bash run_wgeo_ablation.sh
#
# 注意：本脚本会重复运行 VGGT 推理（每组系数生成新的 W.npy）+ 3DGS 训练 + 评估

set -e

PY_VGGT="D:/miniconda/envs/vggt/python.exe"
PY_3DGS="D:/miniconda/envs/gaussian_splatting/python.exe"
SCENE_DIR="d:/Downloads/研究/vggt/examples/llff_fern"
EXPORT_SCRIPT="d:/Downloads/研究/vggt/export_vggt_for_3dgs.py"
CONVERT_SCRIPT="d:/Downloads/研究/vggt/convert_vggt_to_3dgs.py"
TRAIN_SCRIPT="d:/Downloads/研究/gaussian-splatting-main/train.py"
OUT_BASE="d:/Downloads/研究/output/ablation_wgeo"

mkdir -p "$OUT_BASE"

# 5 组系数（w_sigma, w_vis, w_cons）
declare -a CONFIGS=(
    "0.5 0.3 0.2"   # 默认（当前最优经验）
    "0.4 0.3 0.3"   # 一致性更重要
    "0.6 0.2 0.2"   # 深度可靠性更主导
    "0.33 0.33 0.33" # 三分量均权
    "0.7 0.2 0.1"   # 偏向深度可靠性
)

echo "=== W_geo 消融实验（fern 场景，5 组系数）==="
echo ""

for config in "${CONFIGS[@]}"; do
    read -ra cs <<< "$config"
    ws="${cs[0]}"
    wv="${cs[1]}"
    wc="${cs[2]}"
    tag="ws${ws}_wv${wv}_wc${wc}"
    echo "--------------------------------"
    echo "[Config] w_sigma=$ws w_vis=$wv w_cons=$wc"
    echo "--------------------------------"

    # 1. 重跑 VGGT 推理（重新生成 W.npy，因为系数会变）
    # 注：VGGT 推理本身的输出（depth、tracks）不变，只是 W.npy 会按系数重新计算
    # 优化：单独写个脚本只重算 W.npy 而不重跑整个 VGGT 推理
    # 这里暂时全跑，不优化
    "$PY_VGGT" -u -X utf8 "$EXPORT_SCRIPT" \
        --scene_dir "$SCENE_DIR" \
        --num_frames 8 \
        --w_sigma "$ws" --w_vis "$wv" --w_cons "$wc" \
        --w_strategy weighted_avg --w_min 0.1 \
        2>&1 | tail -3

    # 2. 重跑 convert（生成 3dgs_input + train/test split）
    rm -rf "$SCENE_DIR/3dgs_input"
    "$PY_3DGS" "$CONVERT_SCRIPT" \
        --vggt_dir "$SCENE_DIR/vggt" \
        --output_dir "$SCENE_DIR/3dgs_input" \
        2>&1 | tail -3

    # 3. 训练 3DGS（带 --eval）
    OUT_DIR="$OUT_BASE/fern_${tag}"
    rm -rf "$OUT_DIR"
    "$PY_3DGS" "$TRAIN_SCRIPT" \
        -s "$SCENE_DIR/3dgs_input" \
        -d depths --weights weights --eval \
        -m "$OUT_DIR" \
        --iterations 7000 --test_iterations 7000 --save_iterations 7000 \
        --disable_viewer \
        2>&1 | grep "ITER 7000"

    # 4. 渲染 + 评估
    "$PY_3DGS" "d:/Downloads/研究/gaussian-splatting-main/render.py" \
        -m "$OUT_DIR" --iteration 7000 2>&1 | tail -1

    "$PY_3DGS" "d:/Downloads/研究/gaussian-splatting-main/metrics_train.py" \
        -m "$OUT_DIR" 2>&1 | tail -3

    echo ""
done

echo "=== 消融实验完成！结果在 $OUT_BASE ==="
