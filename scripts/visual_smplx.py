#!/usr/bin/env python3
"""Visualize SMPL-X parameters stored in a pickle dict."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import pickle

import numpy as np
import torch

from detrsmpl.core.visualization.render_smpl_headless import render_smpl_headless
from detrsmpl.models.body_models.builder import build_body_model


DEFAULT_RENDER_RESOLUTION = (720, 720)
DEFAULT_RENDER_CHOICE = "mq"
DEFAULT_PALETTE = np.array((0.7, 0.7, 0.7), dtype=np.float32)
DEFAULT_ORBIT_SPEED = 0.0


def _render_sequence(
    verts,
    *,
    body_model,
    device,
    output_path,
    resolution=DEFAULT_RENDER_RESOLUTION,
    render_choice=DEFAULT_RENDER_CHOICE,
    palette=DEFAULT_PALETTE,
    orbit_speed=DEFAULT_ORBIT_SPEED,
    alpha=1.0,
) -> None:
    """Render a SMPL-X sequence to disk without requiring input imagery.

    This wraps :func:`render_smpl` with pre-configured defaults tuned for
    quick visual inspection of pickle results.
    """
    if isinstance(verts, torch.Tensor):
        verts_data = verts.detach().cpu().numpy()
    else:
        verts_data = np.asarray(verts)

    render_smpl_headless(
        verts=verts_data,
        body_model=body_model,
        output_path=output_path,
        device=device,
        resolution=resolution,
        render_choice=render_choice,
        palette=palette,
        orbit_speed=orbit_speed,
        alpha=alpha,
        overwrite=True,
    )


def _apply_final_rotation(vertices: torch.Tensor) -> torch.Tensor:
    """Rotate vertices 180° around Z followed by Y to fix the final orientation."""
    rot_z = vertices.new_tensor(
        (
            (-1.0, 0.0, 0.0),
            (0.0, -1.0, 0.0),
            (0.0, 0.0, 1.0),
        )
    )
    rot_y = vertices.new_tensor(
        (
            (-1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, -1.0),
        )
    )
    rotation = rot_y @ rot_z
    return vertices @ rotation.t()


def _smooth_vertices_sequence(verts_stack: np.ndarray) -> np.ndarray:
    """Apply a light temporal smoothing filter to reduce animation jitter."""
    if verts_stack.shape[0] < 3:
        return verts_stack

    padded = np.pad(verts_stack, ((1, 1), (0, 0), (0, 0)), mode="edge")
    kernel = np.array((0.25, 0.5, 0.25), dtype=verts_stack.dtype)
    smoothed = (
        kernel[0] * padded[:-2]
        + kernel[1] * padded[1:-1]
        + kernel[2] * padded[2:]
    )
    return smoothed


def _focus_upper_body_view(verts_stack: np.ndarray, focus_ratio: float = 0.25, zoom: float = 1.4) -> np.ndarray:
    """Recenter and zoom vertices so renders emphasize the upper body."""
    if verts_stack.size == 0:
        return verts_stack

    focused = verts_stack.copy()
    xy_center = focused[..., :2].mean(axis=1, keepdims=True)
    focused[..., :2] = (focused[..., :2] - xy_center) * zoom

    z_values = focused[..., 2]
    z_min = z_values.min(axis=1, keepdims=True)
    z_max = z_values.max(axis=1, keepdims=True)
    height = np.maximum(z_max - z_min, 1e-6)
    target_center = z_min + focus_ratio * height
    focused[..., 2] = (z_values - target_center) * zoom

    return focused


def _ensure_tensor(array, length):
    value = torch.as_tensor(array, dtype=torch.float32).view(-1)
    if value.numel() != length:
        raise ValueError(f"Expected {length} values, got {value.numel()}")
    return value.view(1, length)


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize SMPL-X params from .pkl")
    parser.add_argument("pkl_file", type=Path, help="Pickle file containing the SMPL-X dict")
    parser.add_argument("--device", default="cpu", help="cpu or cuda:*")
    parser.add_argument("--gender", choices=(-1, 0, 1), type=int, default=-1,
                        help="-1: neutral, 0: male, 1: female")
    parser.add_argument("--save-mesh", type=Path, default=None,
                        help="Optional path to write an OBJ mesh")
    return parser.parse_args()


def _extract_mesh(params, body_model, device):
    global_orient = _ensure_tensor(params["smplx_root_pose"], 3)
    body_pose = _ensure_tensor(params["smplx_body_pose"], 63)
    left_hand_pose = _ensure_tensor(params["smplx_lhand_pose"], 45)
    right_hand_pose = _ensure_tensor(params["smplx_rhand_pose"], 45)
    jaw_pose = _ensure_tensor(params["smplx_jaw_pose"], 3)
    betas = _ensure_tensor(params["smplx_shape"], 10)
    expression = _ensure_tensor(params["smplx_expr"], 10)
    transl = torch.zeros((1, 3), dtype=torch.float32)

    with torch.no_grad():
        output = body_model(
            betas=betas.to(device),
            global_orient=global_orient.to(device),
            body_pose=body_pose.to(device),
            left_hand_pose=left_hand_pose.to(device),
            right_hand_pose=right_hand_pose.to(device),
            jaw_pose=jaw_pose.to(device),
            expression=expression.to(device),
            transl=transl.to(device),
        )

    vertices = output["vertices"][0].cpu()
    joints = output["joints"][0].cpu()
    return vertices, joints


def _write_mesh(mesh_path, vertices, faces):
    mesh_path.parent.mkdir(parents=True, exist_ok=True)
    with open(mesh_path, "w", encoding="ascii") as obj:
        for v in vertices:
            obj.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for f in faces:
            obj.write(f"f {f[0] + 1} {f[1] + 1} {f[2] + 1}\n")
    print(f"Mesh written to {mesh_path}")


def main():
    args = parse_args()
    input_path = args.pkl_file

    if input_path.is_dir():
        pkl_files = sorted(p for p in input_path.glob('*.pkl') if p.is_file())
        if not pkl_files:
            raise FileNotFoundError(f"No .pkl files found in {input_path}")
        base_name = input_path.name
    else:
        if not input_path.is_file():
            raise FileNotFoundError(f"{input_path} does not exist")
        pkl_files = [input_path]
        base_name = input_path.stem

    if args.save_mesh:
        base_dir = args.save_mesh.parent if args.save_mesh.suffix else args.save_mesh
    else:
        base_dir = Path.cwd()

    base_dir = base_dir if base_dir.is_absolute() else Path.cwd() / base_dir
    output_root = base_dir / base_name
    output_root.mkdir(parents=True, exist_ok=True)

    mesh_targets = {p: output_root / f"{p.stem}.obj" for p in pkl_files}

    body_model_cfg = dict(
        type='smplx',
        keypoint_src='smplx',
        keypoint_dst='smplx',
        model_path='data/body_models/smplx',
        num_expression_coeffs=10,
        num_betas=10,
        use_pca=False,
        use_face_contour=True,
        gender={-1: 'neutral', 0: 'male', 1: 'female'}[args.gender],
    )

    device = torch.device(args.device)
    body_model = build_body_model(body_model_cfg).to(device).eval()

    faces = getattr(body_model, 'faces_tensor', None)
    if faces is None:
        faces = torch.from_numpy(body_model.faces.astype(np.int32))
    faces = faces.cpu().numpy()

    all_vertices = []
    for pkl_path in pkl_files:
        with open(pkl_path, 'rb') as handle:
            params = pickle.load(handle)
        vertices, _ = _extract_mesh(params, body_model, device)
        vertices = _apply_final_rotation(vertices)
        target = mesh_targets[pkl_path]
        if target:
            _write_mesh(target, vertices, faces)
        all_vertices.append(vertices.numpy())

    if not all_vertices:
        return

    verts_stack = np.stack(all_vertices)
    if len(verts_stack) > 1:
        verts_stack = _smooth_vertices_sequence(verts_stack)
    render_stack = _focus_upper_body_view(verts_stack)
    anim_base = input_path.name if input_path.is_dir() else input_path.stem
    anim_dir = mesh_targets[pkl_files[0]].parent
    anim_dir.mkdir(parents=True, exist_ok=True)

    device_label = str(device)
    if len(render_stack) > 1:
        anim_path = anim_dir / f"{anim_base}.mp4"
        _render_sequence(
            render_stack,
            body_model=body_model,
            device=device_label,
            output_path=anim_path,
            orbit_speed=DEFAULT_ORBIT_SPEED,
        )
        print(f"Animation written to {anim_path}")
    else:
        frame_path = anim_dir / f"{anim_base}.png"
        _render_sequence(
            render_stack,
            body_model=body_model,
            device=device_label,
            output_path=frame_path,
            orbit_speed=0.0,
        )
        print(f"Single frame saved to {frame_path}")



if __name__ == "__main__":
    main()