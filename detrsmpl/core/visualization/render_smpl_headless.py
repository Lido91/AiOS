"""Utilities for rendering SMPL/SMPL-X sequences without image backgrounds."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from detrsmpl.core.visualization.visualize_smpl import render_smpl

Number = Union[int, float]
BackgroundColor = Union[Number, Sequence[Number], np.ndarray, torch.Tensor]


def _normalize_device(device: Union[str, torch.device]) -> torch.device:
    if isinstance(device, torch.device):
        return device
    return torch.device(device)


def _normalize_resolution(resolution: Union[int, Iterable[int], None]) -> Tuple[int, int]:
    if resolution is None:
        return (720, 720)
    if isinstance(resolution, int):
        return (resolution, resolution)
    resolution = tuple(int(v) for v in resolution)
    if len(resolution) != 2:
        raise ValueError('resolution must be an int or an iterable of two ints')
    return resolution


def _to_tensor(verts: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
    if isinstance(verts, np.ndarray):
        verts = torch.from_numpy(verts)
    if not isinstance(verts, torch.Tensor):
        raise TypeError(f'Unsupported verts type: {type(verts)!r}')
    verts = verts.to(torch.float32)
    if verts.ndim == 2:
        verts = verts.unsqueeze(0)
    if verts.ndim != 3:
        raise ValueError(f'Expected verts with shape (F, V, 3), got {tuple(verts.shape)}')
    return verts


def _prepare_background(
    num_frames: int,
    resolution: Tuple[int, int],
    background_color: Optional[BackgroundColor],
) -> Optional[np.ndarray]:
    if background_color is None:
        return None
    if isinstance(background_color, torch.Tensor):
        color = background_color.detach().cpu().numpy()
    else:
        color = np.array(background_color, dtype=np.float32)
    if color.size == 1:
        color = np.repeat(color, 3)
    if color.size != 3:
        raise ValueError('background_color must broadcast to RGB values')
    color = np.clip(color, 0.0, 255.0).astype(np.float32)
    h, w = resolution
    background = np.empty((num_frames, h, w, 3), dtype=np.float32)
    background[:] = color.reshape(1, 1, 1, 3)
    return background


def render_smpl_headless(
    verts: Union[np.ndarray, torch.Tensor],
    body_model: torch.nn.Module,
    output_path: Union[str, Path],
    *,
    body_model_config: Optional[dict] = None,
    device: Union[str, torch.device] = 'cpu',
    resolution: Union[int, Iterable[int], None] = (720, 720),
    alpha: float = 1.0,
    orbit_speed: Union[float, Tuple[float, float]] = 0.3,
    render_choice: str = 'mq',
    palette: Union[str, Sequence[str], np.ndarray, torch.Tensor] = 'segmentation',
    batch_size: int = 8,
    overwrite: bool = True,
    background_color: Optional[BackgroundColor] = None,
    convention: str = 'pytorch3d',
    projection: str = 'perspective',
) -> Path:
    """Render a SMPL/SMPL-X sequence without external imagery.

    This is a lightweight wrapper around :func:`render_smpl` that prepares
    dummy backgrounds and sensible defaults for stand-alone rendering.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    device = _normalize_device(device)
    resolution = _normalize_resolution(resolution)
    verts_tensor = _to_tensor(verts)
    background = _prepare_background(verts_tensor.shape[0], resolution, background_color)

    render_smpl(
        verts=verts_tensor,
        body_model=body_model,
        body_model_config=body_model_config,
        device=device,
        output_path=str(output_path),
        resolution=resolution,
        alpha=alpha,
        orbit_speed=orbit_speed,
        render_choice=render_choice,
        palette=palette,
        image_array=background,
        in_ndc=True,
        convention=convention,
        projection=projection,
        overwrite=overwrite,
        batch_size=batch_size,
        no_grad=True,
        return_tensor=False,
    )
    return output_path
