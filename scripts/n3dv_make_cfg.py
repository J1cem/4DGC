#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create paper-aligned 4DGC config for one N3DV scene")
    parser.add_argument("--root", default="/home/ljs/4DGC")
    parser.add_argument("--scene", required=True, help="e.g. coffee_martini")
    parser.add_argument(
        "--exp_name",
        default=None,
        help="experiment id for output/test paths, default: same as scene",
    )
    parser.add_argument("--resolution", type=int, default=2)
    parser.add_argument("--frame_end", type=int, default=300)
    parser.add_argument("--lambda_rd_base", type=float, default=5e-4)
    parser.add_argument("--q", type=int, default=1)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    scene = args.scene
    exp_name = args.exp_name or scene

    cfg = {
        "extent": 0,
        "sh_degree": 1,
        "q": args.q,
        "source_path": str(root / "dataset" / scene / "colmap_0"),
        "model_path": str(root / "outputs" / f"{exp_name}_init"),
        "output_path": str(root / "outputs" / f"{exp_name}_frames"),
        "video_path": str(root / "dataset" / scene),
        "ply_name": "points3D.ply",
        "images": "images",
        "resolution": args.resolution,
        "white_background": False,
        "data_device": "cuda",
        "eval": True,
        "iterations": 400,
        "iterations_s2": 0,
        "first_load_iteration": 10000,
        "position_lr_init": 0.0024,
        "position_lr_final": 2.4e-05,
        "position_lr_delay_mult": 0.01,
        "position_lr_max_steps": 30000,
        "feature_lr": 0.0375,
        "opacity_lr": 0.75,
        "scaling_lr": 0.075,
        "rotation_lr": 0.015,
        "percent_dense": 0.01,
        "lambda_dssim": 0.2,
        "lambda_rd_base": args.lambda_rd_base,
        "depth_smooth": 0.0,
        "lambda_dxyz": 0,
        "lambda_drot": 0,
        "densification_interval": 20,
        "opacity_reset_interval": 3000,
        "densify_from_iter": 380,
        "densify_until_iter": 15000,
        "densify_grad_threshold": 0.00015,
        "mem_path": str(root / "mem" / f"{exp_name}.pth"),
        "batch_size": 1,
        "s2_adding": False,
        "num_of_split": 1,
        "num_of_spawn": 1,
        "std_scale": 2,
        "min_opacity": 0.01,
        "rotate_sh": False,
        "only_mlp": False,
        "convert_SHs_python": False,
        "compute_cov3D_python": False,
        "debug": False,
        "bwd_depth": False,
        "opt_type": "4DGC",
        "ip": "127.0.0.1",
        "port": 6009,
        "debug_from": -1,
        "detect_anomaly": False,
        "test_iterations": [400],
        "save_iterations": [400],
        "frame_start": 1,
        "frame_end": args.frame_end,
        "quiet": False,
        "checkpoint_iterations": [],
        "start_checkpoint": None,
        "read_config": True,
        "load_iteration": 400,
    }

    out_dir = root / "test" / exp_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "cfg_args.json"
    out_file.write_text(json.dumps(cfg, indent=4), encoding="utf-8")
    print(f"Wrote config: {out_file}")


if __name__ == "__main__":
    main()
