#!/usr/bin/env python3
import argparse
from pathlib import Path

import pycolmap


def main() -> None:
    parser = argparse.ArgumentParser(description="Check colmap_0 sparse quality")
    parser.add_argument("--scene_dir", required=True)
    args = parser.parse_args()

    scene_dir = Path(args.scene_dir).resolve()
    sparse = scene_dir / "colmap_0" / "sparse" / "0"
    if not sparse.is_dir():
        raise SystemExit(f"Missing sparse model: {sparse}")

    rec = pycolmap.Reconstruction(str(sparse))
    errs = [p.error for p in rec.points3D.values() if p.error == p.error]
    cams = {img.camera_id for img in rec.images.values()}

    print(f"scene: {scene_dir.name}")
    print(f"registered images: {rec.num_reg_images()}")
    print(f"points3D: {rec.num_points3D()}")
    print(f"camera ids used: {len(cams)}")
    if errs:
        print(f"reproj error mean: {sum(errs)/len(errs):.4f}")
        print(f"reproj error median: {sorted(errs)[len(errs)//2]:.4f}")
    else:
        print("reproj error: n/a")

    if rec.num_reg_images() < 10:
        print("[WARN] Too few registered images.")
    if rec.num_points3D() < 2000:
        print("[WARN] Too few sparse points.")


if __name__ == "__main__":
    main()
