#!/usr/bin/env python3
import argparse
import shutil
from pathlib import Path

import pycolmap


def _report(model_dir: Path) -> None:
    rec = pycolmap.Reconstruction(str(model_dir))
    errs = [p.error for p in rec.points3D.values() if p.error == p.error]
    cam_ids = {img.camera_id for img in rec.images.values()}
    mean_err = (sum(errs) / len(errs)) if errs else float("nan")
    med_err = sorted(errs)[len(errs) // 2] if errs else float("nan")

    print("[REPORT]")
    print(f"  registered images: {rec.num_reg_images()}")
    print(f"  points3D:          {rec.num_points3D()}")
    print(f"  cameras used:      {len(cam_ids)}")
    print(f"  reproj mean/med:   {mean_err:.4f} / {med_err:.4f}")



def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run COLMAP SfM on colmap_0/images using pycolmap."
    )
    parser.add_argument("--scene_dir", required=True, help="e.g. dataset/coffee_martini")
    parser.add_argument("--frame", type=int, default=0, help="colmap_<frame>")
    parser.add_argument(
        "--camera_model",
        default="SIMPLE_PINHOLE",
        choices=["SIMPLE_PINHOLE", "PINHOLE"],
    )
    parser.add_argument(
        "--camera_mode",
        default="SINGLE",
        choices=["AUTO", "SINGLE", "PER_FOLDER", "PER_IMAGE"],
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda", "auto"],
    )
    parser.add_argument("--keep_existing_db", action="store_true")
    args = parser.parse_args()

    scene_dir = Path(args.scene_dir).resolve()
    frame_dir = scene_dir / f"colmap_{args.frame}"
    image_path = frame_dir / "images"
    db_path = frame_dir / "database.db"
    sparse_path = frame_dir / "sparse"
    sparse0 = sparse_path / "0"

    if not image_path.is_dir():
        raise SystemExit(f"images directory not found: {image_path}")

    if db_path.exists() and not args.keep_existing_db:
        db_path.unlink()

    sparse_path.mkdir(parents=True, exist_ok=True)
    if sparse0.exists():
        shutil.rmtree(sparse0)
    for p in sparse_path.iterdir():
        if p.is_dir() and p.name.isdigit():
            shutil.rmtree(p)

    camera_mode = getattr(pycolmap.CameraMode, args.camera_mode)
    device = getattr(pycolmap.Device, args.device)

    print("[1/3] Feature extraction")
    pycolmap.extract_features(
        database_path=str(db_path),
        image_path=str(image_path),
        camera_mode=camera_mode,
        camera_model=args.camera_model,
        device=device,
    )

    print("[2/3] Exhaustive matching")
    pycolmap.match_exhaustive(database_path=str(db_path), device=device)

    print("[3/3] Incremental mapping")
    maps = pycolmap.incremental_mapping(
        database_path=str(db_path),
        image_path=str(image_path),
        output_path=str(sparse_path),
    )

    if len(maps) == 0:
        raise SystemExit("No valid model was reconstructed.")

    best_id = max(
        maps.keys(),
        key=lambda k: (maps[k].num_reg_images(), maps[k].num_points3D()),
    )
    best_dir = sparse_path / str(best_id)
    if not best_dir.exists():
        raise SystemExit(f"Best model directory does not exist: {best_dir}")

    # If best_id is already 0, keep it in-place and avoid deleting the source.
    if best_id != 0:
        if sparse0.exists():
            shutil.rmtree(sparse0)
        shutil.copytree(best_dir, sparse0)

    print(f"Done. Best model id: {best_id}")
    print(f"Output: {sparse0}")
    _report(sparse0)


if __name__ == "__main__":
    main()
