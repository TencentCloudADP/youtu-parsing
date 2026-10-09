"""Image preprocessing utilities for VITA OmniEncoder plugin.

Implements the same patchify logic as the Youtu-Parsing-Omni image processor:
1. smart_resize to multiple of (patch_size * spatial_merge_size)
2. Normalize: (pixel / 255 - 0.5) / 0.5
3. Patchify with pixel_shuffle ordering (merge-compatible layout)
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
from PIL import Image


# Default values of the Youtu-Parsing-Omni image processor
DEFAULT_PATCH_SIZE = 16
DEFAULT_SPATIAL_MERGE_SIZE = 2
DEFAULT_TEMPORAL_PATCH_SIZE = 1
DEFAULT_MEAN = [0.5, 0.5, 0.5]
DEFAULT_STD = [0.5, 0.5, 0.5]
DEFAULT_MIN_PIXELS = 4096      # ~64x64
DEFAULT_MAX_PIXELS = 16777216  # ~4096x4096


def smart_resize(
    height: int,
    width: int,
    factor: int,
    min_pixels: int,
    max_pixels: int,
) -> tuple[int, int]:
    """Adapted from ``smart_resize`` in Hugging Face transformers
    (models/qwen2_vl/image_processing_qwen2_vl.py, Apache-2.0).

    Rescales so that:
    1. Both dims are divisible by ``factor``.
    2. Total pixels stay within ``[min_pixels, max_pixels]``.
    3. Aspect ratio is preserved as closely as possible.

    Returns ``(resized_height, resized_width)``.
    """
    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            "absolute aspect ratio must be smaller than 200, got "
            f"{max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def _resize_native(
    image: Image.Image,
    factor: int,
    min_pixels: int,
    max_pixels: int,
) -> Image.Image:
    """Port of YoutuVITAImageProcessor.process_native resize logic.

    IMPORTANT: this must match the native implementation *exactly*, otherwise
    the patch count differs from the reference and vision embeddings misalign.
    (verified against image_processor.py:207-246)

    Unlike ``_resize_pad_native_v2``, this path does NOT pad: it uses
    ``smart_resize`` to directly resize both dims to a multiple of ``factor``
    (distorting the aspect ratio slightly by rounding), then returns the
    resized image with no mean-colored canvas.
    """
    width, height = image.size
    resized_height, resized_width = smart_resize(
        height, width, factor=factor, min_pixels=min_pixels, max_pixels=max_pixels
    )
    return image.resize((resized_width, resized_height), resample=Image.BICUBIC)


def _resize_pad_native_v2(
    image: Image.Image,
    factor: int,
    mean: list[float],
    min_pixels: int,
    max_pixels: int,
) -> Image.Image:
    """Port of YoutuVITAImageProcessor.process_native_v2 resize+pad logic.

    IMPORTANT: this must match the native implementation *exactly*, otherwise
    the patch count differs from the reference and vision embeddings misalign.
    (verified against image_processing_youtu_vita.py:343-422)

    Steps:
    1. Resize keeping aspect ratio so total pixels fall within [min, max].
    2. Pad (ceil) both dims to a multiple of ``factor``; if the padded canvas
       overflows max_pixels, pad down (floor) and shrink the content to fit.
    3. Paste the resized content centered onto a mean-colored canvas.
    """
    width, height = image.size

    # Step 1: aspect-ratio resize into [min_pixels, max_pixels]
    resized_height, resized_width = height, width
    cur_pixels = resized_height * resized_width
    if cur_pixels > max_pixels:
        beta = math.sqrt(cur_pixels / max_pixels)
        resized_height = max(1, int(math.floor(height / beta)))
        resized_width = max(1, int(math.floor(width / beta)))
    elif cur_pixels < min_pixels:
        beta = math.sqrt(min_pixels / cur_pixels)
        resized_height = max(1, int(math.ceil(height * beta)))
        resized_width = max(1, int(math.ceil(width * beta)))

    # Step 2: pad up (ceil) to factor multiple; fall back to floor+shrink if over max
    padded_height = math.ceil(resized_height / factor) * factor
    padded_width = math.ceil(resized_width / factor) * factor

    if padded_height * padded_width > max_pixels:
        padded_height = max(factor, math.floor(resized_height / factor) * factor)
        padded_width = max(factor, math.floor(resized_width / factor) * factor)
        scale = min(padded_height / resized_height, padded_width / resized_width)
        resized_height = max(1, min(padded_height, int(round(resized_height * scale))))
        resized_width = max(1, min(padded_width, int(round(resized_width * scale))))

    # Step 3: resize content and paste onto mean-colored canvas
    image = image.resize((resized_width, resized_height), resample=Image.BICUBIC)
    background_color = tuple(int(x * 255) for x in mean)
    canvas = Image.new("RGB", (padded_width, padded_height), background_color)

    # paste_x = (padded_width - resized_width) // 2
    # paste_y = (padded_height - resized_height) // 2
    # canvas.paste(image, (paste_x, paste_y))
    canvas.paste(image, (0, 0))
    return canvas


def process_image(
    image: Image.Image | np.ndarray | str,
    patch_size: int = DEFAULT_PATCH_SIZE,
    spatial_merge_size: int = DEFAULT_SPATIAL_MERGE_SIZE,
    temporal_patch_size: int = DEFAULT_TEMPORAL_PATCH_SIZE,
    mean: list[float] = DEFAULT_MEAN,
    std: list[float] = DEFAULT_STD,
    min_pixels: int = DEFAULT_MIN_PIXELS,
    max_pixels: int = DEFAULT_MAX_PIXELS,
) -> dict[str, torch.Tensor]:
    """Process a single image into patches.

    Mirrors YoutuVITAImageProcessor.process_native +
    convert_image_to_patches_with_pixel_shuffle so the pixel_values / grid
    match the reference exactly.

    Returns:
        pixel_values: [N_patches, patch_dim] where patch_dim = C * patch² * temporal_patch
        image_grid_thw: [1, 3] = [[T=1, H_patches, W_patches]]
    """
    # Load image
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image).convert("RGB")
    elif isinstance(image, Image.Image):
        image = image.convert("RGB")

    factor = patch_size * spatial_merge_size

    canvas = _resize_native(image, factor, min_pixels, max_pixels)
    padded_width, padded_height = canvas.size

    # Convert to float and normalize
    img_array = np.array(canvas, dtype=np.float32) / 255.0
    mean_arr = np.array(mean, dtype=np.float32)
    std_arr = np.array(std, dtype=np.float32)
    img_array = (img_array - mean_arr) / std_arr

    # To tensor [C, H, W]
    img_tensor = torch.tensor(img_array, dtype=torch.float32).permute(2, 0, 1)

    # Patchify with pixel_shuffle merge ordering
    pixel_values = _convert_image_to_patches_with_pixel_shuffle(
        img_tensor, patch_size, spatial_merge_size, temporal_patch_size)

    # Grid info (padded dims are exact multiples of patch_size)
    grid_h = padded_height // patch_size
    grid_w = padded_width // patch_size
    grid_thw = torch.tensor([[1, grid_h, grid_w]], dtype=torch.long)

    return {
        "pixel_values": pixel_values,
        "image_grid_thw": grid_thw,
    }


def _convert_image_to_patches_with_pixel_shuffle(
    image: torch.Tensor,
    patch_size: int = 16,
    spatial_merge_size: int = 2,
    temporal_patch_size: int = 1,
) -> torch.Tensor:
    """Convert image to patches with merge-compatible ordering.

    Same patch layout as the Youtu-Parsing-Omni image processor (pixel shuffle).

    Args:
        image: [C, H, W] normalized image tensor

    Returns:
        patches: [N_patches, C * temporal_patch * patch² ] flattened patches
    """
    patches = image.unsqueeze(0)  # [1, C, H, W] as (T=1, C, H, W)
    _, channel, image_height, image_width = patches.shape

    grid_t = patches.shape[0] // temporal_patch_size
    grid_h = image_height // patch_size
    grid_w = image_width // patch_size

    patches = patches.reshape(
        grid_t,
        temporal_patch_size,
        channel,
        grid_h // spatial_merge_size,
        spatial_merge_size,
        patch_size,
        grid_w // spatial_merge_size,
        spatial_merge_size,
        patch_size,
    )
    # Permute to merge-compatible layout:
    # (grid_t, grid_h//merge, grid_w//merge, merge_h, merge_w, C, temporal, patch_h, patch_w)
    patches = patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
    flatten_patches = patches.reshape(
        grid_t * grid_h * grid_w,
        channel * temporal_patch_size * patch_size * patch_size,
    )
    return flatten_patches


def compute_image_tokens(grid_thw: torch.Tensor, spatial_merge_size: int = 2) -> int:
    """Compute number of tokens after merger for a given grid.

    tokens = T * (H / merge) * (W / merge)
    """
    t, h, w = int(grid_thw[0, 0]), int(grid_thw[0, 1]), int(grid_thw[0, 2])
    return t * (h // spatial_merge_size) * (w // spatial_merge_size)
