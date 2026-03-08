#!/usr/bin/env python3
import argparse
import glob
import os
import shutil
import subprocess
import sys
from pathlib import Path


def run(cmd):
    subprocess.run(cmd, check=True)


def extract_frames(video_path: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(out_dir / "%06d.png")
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video_path),
        "-start_number",
        "1",
        pattern,
    ]
    run(cmd)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare multi-view mp4 videos into 4DGC colmap_x/images layout"
    )
    parser.add_argument("--scene_dir", required=True, help="e.g. dataset/coffee_martini")
    parser.add_argument("--video_glob", default="cam*.mp4")
    parser.add_argument("--tmp_dir", default="_extract_tmp")
    parser.add_argument("--keep_tmp", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    scene_dir = Path(args.scene_dir).resolve()
    if not scene_dir.is_dir():
        raise SystemExit(f"Scene directory not found: {scene_dir}")
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg not found in PATH.")

    videos = sorted(scene_dir.glob(args.video_glob))
    if not videos:
        raise SystemExit(f"No videos matched '{args.video_glob}' in {scene_dir}")

    existing_colmap = sorted(scene_dir.glob("colmap_*"))
    if existing_colmap and not args.overwrite:
        raise SystemExit(
            f"Found existing colmap_* directories in {scene_dir}. "
            "Use --overwrite to rebuild."
        )

    if args.overwrite:
        for p in existing_colmap:
            if p.is_dir():
                shutil.rmtree(p)

    tmp_root = scene_dir / args.tmp_dir
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    tmp_root.mkdir(parents=True, exist_ok=True)

    print(f"[1/3] Extracting frames from {len(videos)} videos...")
    frame_counts = []
    cam_names = []
    for i, video in enumerate(videos, 1):
        cam_name = video.stem
        cam_names.append(cam_name)
        out_dir = tmp_root / cam_name
        print(f"  [{i}/{len(videos)}] {video.name}")
        extract_frames(video, out_dir)
        count = len(list(out_dir.glob("*.png")))
        if count == 0:
            raise SystemExit(f"No frames extracted from {video}")
        frame_counts.append(count)

    nframes = min(frame_counts)
    if len(set(frame_counts)) != 1:
        print(
            f"[WARN] Frame count mismatch per camera: {frame_counts}. "
            f"Using min frame count {nframes}."
        )

    print(f"[2/3] Creating colmap_0..colmap_{nframes - 1} directories...")
    for fid in range(nframes):
        (scene_dir / f"colmap_{fid}" / "images").mkdir(parents=True, exist_ok=True)

    print("[3/3] Moving extracted images into frame folders...")
    for cam_name in cam_names:
        cam_dir = tmp_root / cam_name
        for fid in range(nframes):
            src = cam_dir / f"{fid + 1:06d}.png"
            dst = scene_dir / f"colmap_{fid}" / "images" / f"{cam_name}.png"
            if not src.exists():
                raise SystemExit(f"Missing frame: {src}")
            os.replace(src, dst)

    if not args.keep_tmp:
        shutil.rmtree(tmp_root, ignore_errors=True)

    print("Done.")
    print(f"Scene path: {scene_dir}")
    print(f"Frames prepared: {nframes}")
    print(f"Cameras prepared: {len(cam_names)} ({', '.join(cam_names)})")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {e}", file=sys.stderr)
        sys.exit(e.returncode)
