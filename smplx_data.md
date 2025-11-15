# SMPL-X Inference Output

<!-- ## Running the pipeline
- Command `scripts/inference.sh <checkpoint> <input_path> <output_label> [num_person] [score_threshold] [gpu_per_node]` launches distributed inference via `torch.distributed.launch`.
- For the How2Sign batch, the command `scripts/inference.sh data/checkpoint/aios_checkpoint.pth /data/hwu/how2sign/how2sign_images_test /data/hwu/how2sign/how2sign_smplx_test 1 0.6 2` writes to `demo//data/hwu/how2sign/how2sign_smplx_test` (note the script prefixes `demo/` even when an absolute output path is supplied).
- GPUs are fixed inside the script by `export CUDA_VISIBLE_DEVICES=2,3`; edit the script if you need a different device list.
- The dataset wrapper supports image folders and `.mp4` files. Videos are first unpacked into frames under a transient `temp_img/` directory inside the output tree.

## Output directory layout
- Root folder: `demo/<output_label>/`.
- Execution metadata dumped alongside the inference results:
  - `config_args_raw.json` – raw CLI arguments captured from `argparse`.
  - `config_args_all.json` – merged configuration after config file + CLI overrides.
  - `config_cfg.py` – final MMDetection-style config snapshot.
  - `info.txt` / `info.txt.rank1` – per-rank logs from the distributed launch showing git state, full command, and selected metrics.
- Results live in `<input_stem>_out/`. For the How2Sign run this is `demo/data/hwu/how2sign/how2sign_smplx_test/how2sign_images_test_out/`.
  - `smplx_params/` mirrors the relative structure of the input dataset. Example: frames from `-fZc293MpJk_0-1-rgb_front` end up under `smplx_params/-fZc293MpJk_0-1-rgb_front/`.
  - Optional folders: `temp_img/` (video extractions) and `mesh/` (placeholder for downstream exports) may appear but are empty unless post-processing scripts populate them. -->

## Frame-wise SMPL-X parameter files
- Each detection that survives the confidence filter is serialized as `frame_XXXXXXXX_person_<idx>.pkl` inside the corresponding sub-directory. Files are written in descending score order until a score falls below `INFERENCE.score_threshold` (currently hard-coded to `0.2`).
- Example file: `demo/data/hwu/how2sign/how2sign_smplx_test/how2sign_images_test_out/smplx_params/-fZc293MpJk_0-1-rgb_front/frame_000004_person_0.pkl`.
- Every pickle contains `float32` arrays shaped `(1, N)` with axis-angle pose parameters (radians):

| Key | Shape | Description |
| --- | ----- | ----------- |
| `cam_trans` | (1, 3) | Weak-perspective camera `(scale, tx, ty)` used by `project_points_new` for projecting model joints back to the image plane. |
| `smplx_root_pose` | (1, 3) | Global body orientation (axis-angle). |
| `smplx_body_pose` | (1, 63) | 21 body joints × 3-axis angles. |
| `smplx_lhand_pose` | (1, 45) | 15 left-hand joints × 3-axis angles. |
| `smplx_rhand_pose` | (1, 45) | 15 right-hand joints × 3-axis angles. |
| `smplx_jaw_pose` | (1, 3) | Jaw rotation (axis-angle). |
| `smplx_expr` | (1, 10) | SMPL-X expression blendshape coefficients. |
| `smplx_shape` | (1, 10) | SMPL-X shape (β) coefficients. |

- Scores are not saved inside the pickle, but the dataset keeps them transiently in-memory to decide which detections to emit.

## Loading and using the data
- Standard Python snippet for inspection:
```python
from pathlib import Path
import pickle

sample = Path("demo/data/hwu/how2sign/how2sign_smplx_test/how2sign_images_test_out/smplx_params/-fZc293MpJk_0-1-rgb_front/frame_000004_person_0.pkl")
with sample.open("rb") as f:
    params = pickle.load(f)
root_pose = params["smplx_root_pose"].reshape(-1, 3)
```
- Body, hand, and jaw poses are axis-angle; reshape to `(n_joints, 3)` before converting to rotation matrices or applying to a SMPL-X body model.
- Camera parameters follow the convention used in `detrsmpl/utils/geometry.py`: `project_points_new` expects `(scale, tx, ty)`, so depth ≈ `1 / scale` and in-plane offsets should be divided by `scale` before use in perspective projection utilities.
- To render meshes or videos from these parameters, use `scripts/visual_smplx.py` (`python scripts/visual_smplx.py --help`) which can read the same pickle structure and output `.obj` sequences or MP4 renders.

## Practical notes
- Multi-person predictions: the script parameter `num_person` controls the number of query slots reserved during inference; within each frame `_person_<idx>` reflects the ranking by confidence.
- Confidence thresholds: the CLI `threshold` override adjusts filtering inside the model’s post-process stage, while the dataset enforces an additional `0.2` cutoff when writing files. Raise or lower this limit in `datasets/INFERENCE.py` if you need more or fewer detections saved.
- When running on new data, keep the input directory tree stable; the output tree will mirror it, simplifying downstream joins with the original frames.
- Check `info.txt` for runtime metadata (git sha, command line, total timing) whenever auditing or re-running experiments.
