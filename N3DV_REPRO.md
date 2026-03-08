# 4DGC N3DV Reproduction (Paper-Aligned)

This guide uses the 4DGC paper settings on N3DV:
- Stage-1 iterations: `400`
- Stage-2 iterations: `100`
- `lambda_dssim = 0.2`
- one held-out camera for test (`eval=true`)

## 1) Prepare one N3DV scene

```bash
# Option A: auto download one supported scene
bash scripts/download_n3dv_scene.sh coffee_martini

# Option B: use local zip
bash scripts/prepare_n3dv_scene.sh coffee_martini --zip /path/to/coffee_martini.zip

# If already extracted, just prepare/check
bash scripts/prepare_n3dv_scene.sh coffee_martini
```

## 2) (Optional) Rebuild COLMAP with fixed-camera prior

```bash
# Rebuild colmap_0 sparse model (recommended for poor generalization)
bash scripts/prepare_n3dv_scene.sh coffee_martini --overwrite --camera-mode SINGLE --camera-model SIMPLE_PINHOLE
```

Quick quality check:

```bash
source .venv310/bin/activate
python scripts/check_colmap0.py --scene_dir dataset/coffee_martini
```

## 3) Generate / adjust config

```bash
source .venv310/bin/activate
python scripts/n3dv_make_cfg.py --scene coffee_martini --exp_name coffee_martini_l0p0005 --lambda_rd_base 0.0005
```

Paper reports RD trade-off with different `lambda1` values. You can sweep:
- `0.0005`, `0.001`, `0.002`, `0.004`, `0.008`

One-command sweep:

```bash
bash scripts/run_n3dv_lambda_sweep.sh coffee_martini
```

## 4) Run full pipeline with resume

```bash
bash scripts/run_n3dv_pipeline.sh coffee_martini coffee_martini_l0p0005
```

The script automatically resumes from:
- `outputs/<scene>_init/chkpnt_latest.pth`
- existing frame outputs in `outputs/<scene>_frames/colmap_*`

## 5) Batch mode (5 common N3DV scenes)

```bash
bash scripts/run_n3dv_all.sh
```

This loops:
- `coffee_martini`
- `cook_spinach`
- `cut_roasted_beef`
- `flame_steak`
- `sear_steak`

`flame_salmon` is split in release assets and should be prepared manually first.
