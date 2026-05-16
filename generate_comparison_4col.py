"""
仅前四列对比图：GT | Baseline | W_geo | W_geo+W_appear (precomp)
输出目录：comparison_visuals_4col/<scene>/
"""
from pathlib import Path

from generate_full_comparison import (
    FRAMES,
    SCENES,
    imwrite_unicode,
    make_comparison_full,
)

# 仅保留 Baseline、W_geo、W_geo+W_app(pre) 三列渲染（加上 GT 共四列）
NUM_METHODS_AFTER_GT = 3

VIS_DIR_4COL = Path(__file__).resolve().parent / "comparison_visuals_4col"


def main():
    VIS_DIR_4COL.mkdir(exist_ok=True)
    for scene_key, methods in SCENES.items():
        subset = methods[:NUM_METHODS_AFTER_GT]
        scene_out = VIS_DIR_4COL / scene_key
        scene_out.mkdir(exist_ok=True)
        print(f"\n=== {scene_key} (4 columns) ===")
        for frame_name in FRAMES:
            comp = make_comparison_full(scene_key, subset, frame_name)
            if comp is None:
                print(f"  skip {frame_name} (missing GT or paths)")
                continue
            out_path = scene_out / f"compare_{frame_name}"
            imwrite_unicode(out_path, comp)
            print(f"  saved: {out_path}")
    print(f"\nDone. Output: {VIS_DIR_4COL}")


if __name__ == "__main__":
    main()
