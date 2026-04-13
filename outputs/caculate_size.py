import argparse
import csv
from pathlib import Path


def collect_metrics(frame_dir: Path, fps: float):
    render_png_kb = 0.0
    for png in frame_dir.glob("*_rendering*.png"):
        render_png_kb += png.stat().st_size / 1024.0

    checkpoint_size_kb = 0.0
    ckpt_files = list(frame_dir.glob("chkpnt*.pth"))
    if ckpt_files:
        checkpoint_size_kb = max(f.stat().st_size for f in ckpt_files) / 1024.0

    model_artifact_size_kb = 0.0
    entropy_bitstream_size_kb = 0.0
    point_cloud_root = frame_dir / "point_cloud"
    if point_cloud_root.exists():
        iter_dirs = sorted(
            [p for p in point_cloud_root.glob("iteration_*") if p.is_dir()],
            key=lambda p: int(p.name.split("_")[-1]),
        )
        if iter_dirs:
            latest_iter = iter_dirs[-1]
            model_artifact_size_kb = sum(f.stat().st_size for f in latest_iter.rglob("*.ply")) / 1024.0
            bitstream_bytes = sum(f.stat().st_size for f in latest_iter.rglob("feature*") if f.is_file())
            entropy_bitstream_size_kb = bitstream_bytes / 1024.0
            real_bitrate_kbps = (bitstream_bytes * 8.0 * fps) / 1000.0 if fps > 0 else 0.0
        else:
            real_bitrate_kbps = 0.0
    else:
        real_bitrate_kbps = 0.0

    return {
        "frame": frame_dir.name,
        "render_png_size_kb": render_png_kb,
        "checkpoint_size_kb": checkpoint_size_kb,
        "model_artifact_size_kb": model_artifact_size_kb,
        "entropy_bitstream_size_kb": entropy_bitstream_size_kb,
        "real_bitrate_kbps": real_bitrate_kbps,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", type=str, required=True, help="Directory containing colmap_* frame outputs.")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--csv_path", type=str, default="")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    frame_dirs = sorted(
        [p for p in output_root.glob("colmap_*") if p.is_dir()],
        key=lambda p: int(p.name.split("_")[-1]),
    )

    rows = [collect_metrics(frame_dir, args.fps) for frame_dir in frame_dirs]
    if not rows:
        print("No frame folders found.")
        return

    csv_path = Path(args.csv_path) if args.csv_path else output_root / "size_metrics.csv"
    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved metrics to {csv_path}")


if __name__ == "__main__":
    main()
