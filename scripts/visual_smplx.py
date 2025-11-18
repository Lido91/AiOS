#!/usr/bin/env python3
"""Visualize SMPL-X parameters stored in a pickle dict."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import pickle

from typing import Sequence, Tuple

import numpy as np
import torch

import imageio.v2 as imageio
from pytorch3d.renderer import (DirectionalLights, FoVPerspectiveCameras,
                               MeshRasterizer, MeshRenderer,
                               RasterizationSettings, SoftPhongShader,
                               TexturesVertex)
from pytorch3d.structures import Meshes

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


def _load_obj_sequence(obj_files: Sequence[Path]) -> Tuple[np.ndarray, np.ndarray]:
    """Load a sequence of OBJ meshes into numpy arrays of verts and faces."""
    vertices_list = []
    faces = None
    for obj_path in obj_files:
        verts = []
        faces_local = []
        with open(obj_path, 'r', encoding='utf-8') as obj_file:
            for raw_line in obj_file:
                line = raw_line.strip()
                if not line or line.startswith('#'):
                    continue
                if line.startswith('v '):
                    parts = line.split()
                    if len(parts) >= 4:
                        verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
                elif line.startswith('f ') and faces is None:
                    tokens = line.split()[1:4]
                    if len(tokens) == 3:
                        faces_local.append([int(tok.split('/')[0]) - 1 for tok in tokens])
        if not verts:
            raise ValueError(f'No vertices found in OBJ file {obj_path}')
        vertices_list.append(np.asarray(verts, dtype=np.float32))
        if faces is None and faces_local:
            faces = np.asarray(faces_local, dtype=np.int32)
    if not vertices_list:
        raise ValueError('No OBJ files provided')
    if faces is None:
        raise ValueError('Faces could not be read from the OBJ sequence')
    verts_stack = np.stack(vertices_list)
    return verts_stack, faces


DEFAULT_OBJ_VIDEO_FPS = 12


def _build_simple_renderer(device: torch.device, resolution: Tuple[int, int]) -> MeshRenderer:
    """Construct a lightweight PyTorch3D renderer for OBJ playback."""
    height, width = int(resolution[0]), int(resolution[1])
    raster_settings = RasterizationSettings(
        image_size=(height, width),
        blur_radius=0.0,
        faces_per_pixel=1,
    )
    cameras = FoVPerspectiveCameras(
        device=device,
        R=torch.eye(3, device=device).unsqueeze(0),
        T=torch.tensor([[0.0, 0.0, 2.5]], device=device),
    )
    lights = DirectionalLights(
        device=device,
        ambient_color=((0.7, 0.7, 0.7),),
        diffuse_color=((0.3, 0.3, 0.3),),
        specular_color=((0.0, 0.0, 0.0),),
        direction=((0.0, 0.0, -1.0),),
    )
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=SoftPhongShader(device=device, cameras=cameras, lights=lights),
    )
    return renderer


def _render_obj_frames(
    verts_stack: np.ndarray,
    faces: np.ndarray,
    *,
    output_path: Path,
    resolution: Tuple[int, int],
    device: str,
    fps: int = DEFAULT_OBJ_VIDEO_FPS,
) -> None:
    """Render a stack of vertices plus faces into an MP4 (and optional PNG)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    render_device = torch.device(device)
    if render_device.type == 'cuda' and not torch.cuda.is_available():
        render_device = torch.device('cpu')
    renderer = _build_simple_renderer(render_device, resolution)

    faces_tensor = torch.as_tensor(faces, dtype=torch.int64, device=render_device).unsqueeze(0)
    num_verts = verts_stack.shape[1]
    vertex_color = torch.full((1, num_verts, 3), 0.7, dtype=torch.float32, device=render_device)

    writer_kwargs = {'fps': fps}
    if output_path.suffix.lower() == '.mp4':
        writer_kwargs['codec'] = 'libx264'
    writer = None
    try:
        writer = imageio.get_writer(output_path, **writer_kwargs)
    except ValueError:
        writer_kwargs.pop('codec', None)
        writer = imageio.get_writer(output_path, **writer_kwargs)

    first_frame = None
    with writer:
        for frame in verts_stack:
            verts_tensor = torch.from_numpy(frame).to(render_device).unsqueeze(0)
            mesh = Meshes(
                verts=verts_tensor,
                faces=faces_tensor,
                textures=TexturesVertex(verts_features=vertex_color),
            )
            image = renderer(mesh)[0, ..., :3].clamp(0.0, 1.0).cpu().numpy()
            frame_u8 = (image * 255.0).astype(np.uint8)
            if first_frame is None:
                first_frame = frame_u8
            writer.append_data(frame_u8)

    if verts_stack.shape[0] == 1 and first_frame is not None:
        imageio.imwrite(output_path.with_suffix('.png'), first_frame)


def _render_obj_directory(
    obj_dir: Path,
    output_root: Path,
    *,
    device: str,
    resolution: Tuple[int, int] = DEFAULT_RENDER_RESOLUTION,
) -> Path:
    obj_files = sorted(p for p in obj_dir.glob('*.obj') if p.is_file())
    if not obj_files:
        raise FileNotFoundError(f'No OBJ files found in {obj_dir}')

    verts_stack, faces = _load_obj_sequence(obj_files)
    if len(verts_stack) > 1:
        verts_stack = _smooth_vertices_sequence(verts_stack)
    render_stack = _focus_upper_body_view(verts_stack)

    video_path = output_root / f"{obj_dir.name}.mp4"
    _render_obj_frames(render_stack, faces, output_path=video_path, resolution=resolution, device=device)
    return video_path


def _ensure_tensor(array, length):
    value = torch.as_tensor(array, dtype=torch.float32).view(-1)
    if value.numel() != length:
        raise ValueError(f"Expected {length} values, got {value.numel()}")
    return value.view(1, length)


def _load_param_container(file_path: Path):
    suffix = file_path.suffix.lower()
    if suffix == '.npz':
        with np.load(file_path, allow_pickle=True) as data:
            return {key: data[key] for key in data.files}
    with open(file_path, 'rb') as handle:
        return pickle.load(handle)


def _is_motion_sequence_container(params) -> bool:
    if not isinstance(params, dict):
        return False
    if 'smplx_body_pose' in params:
        return False
    if 'global_orient' not in params:
        return False
    if 'body_pose_axis' not in params and 'body_pose' not in params:
        return False
    return True


def _normalize_sequence_array(value, num_frames: int, length: int, *, default=None, device=None) -> torch.Tensor:
    if value is None:
        if default is None:
            raise KeyError("Missing required sequence data")
        value = default
    array = np.asarray(value, dtype=np.float32)
    if array.size == 0:
        array = np.zeros((num_frames, length), dtype=np.float32)
    else:
        if array.size % length != 0:
            raise ValueError(f"Incompatible data length {array.size} for expected {length}")
        frames_in_value = array.size // length
        array = array.reshape(frames_in_value, length)
        if frames_in_value == num_frames:
            array = array.copy()
        elif frames_in_value == 1:
            array = np.broadcast_to(array, (num_frames, length)).copy()
        else:
            raise ValueError(f"Unexpected frame count {frames_in_value}, expected {num_frames}")
    tensor = torch.from_numpy(np.ascontiguousarray(array))
    if device is not None:
        tensor = tensor.to(device)
    return tensor


def _extract_sequence_from_motion_params(params, body_model, device):
    num_frames = params.get('num_frames', 0)
    if isinstance(num_frames, (np.integer, int)):
        num_frames = int(num_frames)
    else:
        num_frames = 0
    if num_frames <= 0:
        orient = params.get('global_orient')
        if orient is None:
            raise KeyError("Sequence params missing global_orient")
        orient_array = np.asarray(orient)
        num_frames = orient_array.shape[0] if orient_array.ndim > 1 else 1
    body_pose_key = 'body_pose_axis' if 'body_pose_axis' in params else 'body_pose'
    if body_pose_key is None:
        raise KeyError("Sequence params missing body pose data")

    defaults = {
        'betas': np.zeros((1, 10), dtype=np.float32),
        'expression': np.zeros((1, 10), dtype=np.float32),
        'jaw_pose': np.zeros((1, 3), dtype=np.float32),
        'left_hand_pose': np.zeros((1, 45), dtype=np.float32),
        'right_hand_pose': np.zeros((1, 45), dtype=np.float32),
        'cam_trans': np.zeros((1, 3), dtype=np.float32),
    }

    global_orient = _normalize_sequence_array(params['global_orient'], num_frames, 3, device=device)
    body_pose = _normalize_sequence_array(params[body_pose_key], num_frames, 63, device=device)
    left_hand = _normalize_sequence_array(params.get('left_hand_pose'), num_frames, 45,
                                          default=defaults['left_hand_pose'], device=device)
    right_hand = _normalize_sequence_array(params.get('right_hand_pose'), num_frames, 45,
                                           default=defaults['right_hand_pose'], device=device)
    jaw_pose = _normalize_sequence_array(params.get('jaw_pose'), num_frames, 3,
                                         default=defaults['jaw_pose'], device=device)
    betas = _normalize_sequence_array(params.get('betas'), num_frames, 10,
                                      default=defaults['betas'], device=device)
    expression = _normalize_sequence_array(params.get('expression'), num_frames, 10,
                                           default=defaults['expression'], device=device)
    transl = _normalize_sequence_array(params.get('cam_trans'), num_frames, 3,
                                       default=defaults['cam_trans'], device=device)

    with torch.no_grad():
        output = body_model(
            betas=betas,
            global_orient=global_orient,
            body_pose=body_pose,
            left_hand_pose=left_hand,
            right_hand_pose=right_hand,
            jaw_pose=jaw_pose,
            expression=expression,
            transl=transl,
        )

    vertices = output["vertices"].cpu()
    joints = output["joints"].cpu()
    return vertices, joints


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

    # import pdb
    # pdb.set_trace()

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

    if not input_path.exists():
        raise FileNotFoundError(f"{input_path} does not exist")

    base_name = input_path.name if input_path.is_dir() else input_path.stem

    if args.save_mesh:
        base_dir = args.save_mesh.parent if args.save_mesh.suffix else args.save_mesh
    else:
        base_dir = Path.cwd()

    base_dir = base_dir if base_dir.is_absolute() else Path.cwd() / base_dir
    output_root = base_dir / base_name
    output_root.mkdir(parents=True, exist_ok=True)

    param_cache = {}
    sequence_cache = {}

    if input_path.is_dir():
        pkl_files = sorted(
            p for p in input_path.glob('*')
            if p.is_file() and p.suffix.lower() in ('.pkl', '.npz')
        )
        obj_files = sorted(p for p in input_path.glob('*.obj') if p.is_file())
        if obj_files and not pkl_files:
            video_path = _render_obj_directory(input_path, output_root, device=args.device)
            print(f"Animation written to {video_path}")
            return
        if not pkl_files:
            raise FileNotFoundError(f"No .pkl or .npz files found in {input_path}")
    else:
        pkl_files = [input_path]
        if input_path.suffix.lower() in ('.pkl', '.npz'):
            params = _load_param_container(input_path)
    
            if _is_motion_sequence_container(params):
                sequence_cache[input_path] = params
            else:
                param_cache[input_path] = params

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
        params = sequence_cache.get(pkl_path)
        if params is None:
            params = param_cache.pop(pkl_path, None)
            if params is None:
                params = _load_param_container(pkl_path)
                if _is_motion_sequence_container(params):
                    sequence_cache[pkl_path] = params
        if params is not None and _is_motion_sequence_container(params):
            vertices_batch, _ = _extract_sequence_from_motion_params(params, body_model, device)
            vertices_batch = _apply_final_rotation(vertices_batch)
            target = mesh_targets.get(pkl_path)
            if target:
                _write_mesh(target, vertices_batch[0].numpy(), faces)
            all_vertices.extend(frame.numpy() for frame in vertices_batch)
            continue
        if params is None:
            raise ValueError(f"Failed to load parameters from {pkl_path}")
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
