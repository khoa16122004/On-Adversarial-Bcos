from __future__ import annotations

import torch
import torch.nn.functional as F


def _flatten_index(channel: int, row: int, col: int, image_size: int) -> int:
    return channel * image_size * image_size + row * image_size + col


def diagonal_order(image_size: int, channels: int = 3) -> torch.Tensor:
    indices: list[int] = []
    for diag in range(2 * image_size - 1):
        row_start = max(0, diag - (image_size - 1))
        row_end = min(image_size - 1, diag)
        for row in range(row_start, row_end + 1):
            col = diag - row
            for channel in range(channels):
                indices.append(_flatten_index(channel, row, col, image_size))
    return torch.tensor(indices, dtype=torch.long)


def block_order(
    image_size: int,
    channels: int = 3,
    initial_size: int | None = None,
    stride: int = 1,
) -> torch.Tensor:
    if initial_size is None:
        initial_size = image_size
    initial_size = int(max(1, min(initial_size, image_size)))
    stride = int(max(1, stride))

    visited = torch.zeros((image_size, image_size), dtype=torch.bool)
    indices: list[int] = []

    # Prioritize low-frequency corner first.
    for row in range(initial_size):
        for col in range(initial_size):
            if not visited[row, col]:
                visited[row, col] = True
                for channel in range(channels):
                    indices.append(_flatten_index(channel, row, col, image_size))

    # Then sweep the remaining coordinates in strided blocks.
    for row0 in range(0, image_size, stride):
        for col0 in range(0, image_size, stride):
            row1 = min(image_size, row0 + stride)
            col1 = min(image_size, col0 + stride)
            for row in range(row0, row1):
                for col in range(col0, col1):
                    if not visited[row, col]:
                        visited[row, col] = True
                        for channel in range(channels):
                            indices.append(_flatten_index(channel, row, col, image_size))

    return torch.tensor(indices, dtype=torch.long)


def block_idct(x: torch.Tensor, block_size: int, linf_bound: float = 0.0) -> torch.Tensor:
    # Approximation fallback: smooth upsample from low-frequency grid to image resolution.
    if x.ndim != 4:
        raise ValueError("Expected x with shape [B, C, H, W].")

    out = F.interpolate(x, size=(block_size, block_size), mode="bilinear", align_corners=False)
    if linf_bound > 0:
        out = out.clamp(-float(linf_bound), float(linf_bound))
    return out
