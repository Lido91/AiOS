#!/usr/bin/env python3
"""Render a sequence of OBJ meshes into a headless MP4 preview."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
from typing import Sequence, Tuple

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import imageio.v2 as imageio
except ImportError:
    try:
        import imageio
    except ImportError:
        imageio = None
import numpy as np
import torch
from pytorch3d.renderer import (
    DirectionalLights,
    FoVPerspectiveCameras,
    MeshRasterizer,
    MeshRenderer,
    RasterizationSettings,
    SoftPhongShader,
    TexturesVertex,
)
from pytorch3d.structures import Meshes

DEFAULT_RENDER_RESOLUTION: Tuple[int, int] = (720, 720)
DEFAULT_OBJ_VIDEO_FPS = 24
DEFAULT_VERTEX_COLOR = 0.7
DEFAULT_ZOOM = 1.4
DEFAULT_FOCUS_RATIO = 0.25


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


def _focus_upper_body_view(
    verts_stack: np.ndarray,
    *,
    focus_ratio: float = DEFAULT_FOCUS_RATIO,
    zoom: float = DEFAULT_ZOOM,
) -> np.ndarray:
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



def _apply_final_rotation(verts_stack: np.ndarray) -> np.ndarray:
    """Rotate vertices 180° around Z axis to match SMPL-X output."""
    if verts_stack.size == 0:
        return verts_stack
    rotation = np.array(
        ((-1.0,  0.0, 0.0),
         ( 0.0, -1.0, 0.0),
         ( 0.0,  0.0, 1.0)),
        dtype=verts_stack.dtype
    )
    return verts_stack @ rotation.T

def _load_obj_sequence(obj_files: Sequence[Path]) -> Tuple[np.ndarray, np.ndarray]:
    """Load vertices and faces from a sequence of OBJ files."""
    vertices_list = []
    faces = None
    for obj_path in obj_files:
        verts = []
        faces_local = []
        with open(obj_path, "r", encoding="utf-8") as obj_file:
            for raw_line in obj_file:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("v "):
                    parts = line.split()
                    if len(parts) >= 4:
                        verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
                elif line.startswith("f ") and faces is None:
                    tokens = line.split()[1:]
                    if len(tokens) < 3:
                        continue
                    # fan triangulation for polygons with more than 3 vertices
                    first = tokens[0]
                    for idx in range(1, len(tokens) - 1):
                        tri = (first, tokens[idx], tokens[idx + 1])
                        faces_local.append([
                            int(tok.split("/")[0]) - 1
                            for tok in tri
                        ])
        if not verts:
            raise ValueError(f"No vertices found in OBJ file {obj_path}")
        vertices_list.append(np.asarray(verts, dtype=np.float32))
        if faces is None and faces_local:
            faces = np.asarray(faces_local, dtype=np.int32)
    if not vertices_list:
        raise ValueError("No OBJ files provided")
    if faces is None:
        raise ValueError("Faces could not be read from the OBJ sequence")
    verts_stack = np.stack(vertices_list)
    return verts_stack, faces


def _build_renderer(device: torch.device, resolution: Tuple[int, int]) -> MeshRenderer:
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
    fps: int,
    vertex_color: float,
) -> None:
    """Render a stack of vertices plus faces into a video file (and optional PNG)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    render_device = torch.device(device)
    if render_device.type == "cuda" and not torch.cuda.is_available():
        render_device = torch.device("cpu")
    renderer = _build_renderer(render_device, resolution)

    faces_tensor = torch.as_tensor(faces, dtype=torch.int64, device=render_device).unsqueeze(0)
    num_verts = verts_stack.shape[1]
    vertex_color_tensor = torch.full(
        (1, num_verts, 3),
        float(vertex_color),
        dtype=torch.float32,
        device=render_device,
    )

    frames_rgb = []
    for frame in verts_stack:
        verts_tensor = torch.from_numpy(frame).to(render_device).unsqueeze(0)
        mesh = Meshes(
            verts=verts_tensor,
            faces=faces_tensor,
            textures=TexturesVertex(verts_features=vertex_color_tensor),
        )
        image = renderer(mesh)[0, ..., :3].clamp(0.0, 1.0).cpu().numpy()
        frames_rgb.append((image * 255.0).astype(np.uint8))

    if not frames_rgb:
        return

    height, width = frames_rgb[0].shape[:2]
    suffix = output_path.suffix.lower()

    if cv2 is not None:
        if suffix == '.mp4':
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        elif suffix == '.avi':
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
        else:
            output_path = output_path.with_suffix('.mp4')
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        writer = cv2.VideoWriter(str(output_path), fourcc, float(fps), (width, height))
        if not writer.isOpened():
            raise RuntimeError(f'Failed to open video writer for {output_path}')

        try:
            for frame_rgb in frames_rgb:
                writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
        finally:
            writer.release()

        if len(frames_rgb) == 1:
            cv2.imwrite(
                str(output_path.with_suffix('.png')),
                cv2.cvtColor(frames_rgb[0], cv2.COLOR_RGB2BGR),
            )
    elif imageio is not None:
        writer_kwargs = {"fps": int(fps)}
        if suffix == '.mp4':
            writer_kwargs['codec'] = 'libx264'
        try:
            writer = imageio.get_writer(output_path, **writer_kwargs)
        except ValueError:
            writer_kwargs.pop('codec', None)
            writer = imageio.get_writer(output_path, **writer_kwargs)
        first_frame = None
        with writer:
            for frame_rgb in frames_rgb:
                if first_frame is None:
                    first_frame = frame_rgb
                writer.append_data(frame_rgb)
        if len(frames_rgb) == 1 and first_frame is not None:
            imageio.imwrite(output_path.with_suffix('.png'), first_frame)
    else:
        raise RuntimeError(
            'Neither OpenCV nor imageio is available. Install one of them to enable video writing.'
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render an OBJ sequence to MP4")
    parser.add_argument("obj_dir", type=Path, help="Directory containing OBJ files")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Optional directory to store the rendered video",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Rendering device (cpu or cuda:*)",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        nargs=2,
        metavar=("HEIGHT", "WIDTH"),
        default=None,
        help="Output resolution as height width (defaults to 720 720)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=DEFAULT_OBJ_VIDEO_FPS,
        help="Frames per second for the output video",
    )
    parser.add_argument(
        "--vertex-color",
        type=float,
        default=DEFAULT_VERTEX_COLOR,
        help="Grayscale vertex color in [0, 1]",
    )
    parser.add_argument(
        "--no-smooth",
        action="store_true",
        help="Disable temporal smoothing before rendering",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    obj_dir: Path = args.obj_dir
    if not obj_dir.is_dir():
        raise FileNotFoundError(f"{obj_dir} is not a directory")

    obj_files = sorted(p for p in obj_dir.glob("*.obj") if p.is_file())
    if not obj_files:
        raise FileNotFoundError(f"No OBJ files found in {obj_dir}")

    verts_stack, faces = _load_obj_sequence(obj_files)
    # verts_stack = _apply_final_rotation(verts_stack)
    if not args.no_smooth and len(verts_stack) > 1:
        verts_stack = _smooth_vertices_sequence(verts_stack)
    render_stack = _focus_upper_body_view(verts_stack)

    if args.output_root is not None:
        output_root = args.output_root if args.output_root.is_absolute() else Path.cwd() / args.output_root
    else:
        output_root = Path.cwd() / obj_dir.name
    output_root.mkdir(parents=True, exist_ok=True)

    resolution = tuple(args.resolution) if args.resolution else DEFAULT_RENDER_RESOLUTION
    video_path = output_root / f"{obj_dir.name}.mp4"

    _render_obj_frames(
        render_stack,
        faces,
        output_path=video_path,
        resolution=resolution,
        device=args.device,
        fps=args.fps,
        vertex_color=args.vertex_color,
    )
    print(f"Animation written to {video_path}")


if __name__ == "__main__":
    main()
