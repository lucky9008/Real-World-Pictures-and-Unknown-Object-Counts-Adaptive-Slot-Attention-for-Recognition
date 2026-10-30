"""COCO Annotation Decoding and Mask Processing Module.

Handles polygon segmentation, Run-Length Encoding (RLE), multiple object instances,
mask aggregation, and discrete nearest-neighbor resizing.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw

logger = logging.getLogger(__name__)

# Try to import pycocotools.mask; fall back cleanly if unavailable
try:
    from pycocotools import mask as mask_utils
    _HAS_PYCOCOTOOLS = True
except ImportError:
    mask_utils = None
    _HAS_PYCOCOTOOLS = False


def polygon_to_mask(polygons: List[List[float]], height: int, width: int) -> np.ndarray:
    """Decode polygon coordinates into a binary boolean mask.

    Args:
        polygons: List of polygons, where each polygon is [x1, y1, x2, y2, ...].
        height: Image height.
        width: Image width.

    Returns:
        Binary numpy array of shape (height, width) with dtype bool.
    """
    mask_img = Image.new("1", (width, height), 0)
    draw = ImageDraw.Draw(mask_img)

    for poly in polygons:
        if len(poly) < 6:  # Need at least 3 points (x, y) to form a polygon
            continue
        coords = [(poly[i], poly[i + 1]) for i in range(0, len(poly), 2)]
        draw.polygon(coords, outline=1, fill=1)

    return np.array(mask_img, dtype=bool)


def rle_to_mask(rle_data: Union[Dict[str, Any], List[int]], height: int, width: int) -> np.ndarray:
    """Decode COCO RLE representation into a binary mask.

    Args:
        rle_data: COCO RLE dict or uncompressed counts list.
        height: Image height.
        width: Image width.

    Returns:
        Binary numpy array of shape (height, width) with dtype bool.
    """
    if _HAS_PYCOCOTOOLS and mask_utils is not None:
        if isinstance(rle_data, dict):
            if isinstance(rle_data.get("counts"), list):
                # Uncompressed RLE
                rle = mask_utils.frPyObjects(rle_data, height, width)
            else:
                rle = rle_data
            decoded = mask_utils.decode(rle)
            if decoded.ndim == 3:
                decoded = decoded[:, :, 0]
            return decoded.astype(bool)

    # Pure Python uncompressed RLE fallback
    counts = rle_data["counts"] if isinstance(rle_data, dict) else rle_data
    if isinstance(counts, list):
        flat_mask = np.zeros(height * width, dtype=bool)
        pos = 0
        val = False
        for count in counts:
            flat_mask[pos : pos + count] = val
            pos += count
            val = not val
        return flat_mask.reshape((height, width), order="F")

    logger.warning("RLE format could not be decoded without pycocotools. Returning empty mask.")
    return np.zeros((height, width), dtype=bool)


def decode_coco_annotation(
    annotation: Dict[str, Any], height: int, width: int
) -> np.ndarray:
    """Decode a single COCO annotation record into a binary mask.

    Args:
        annotation: Annotation dictionary from COCO JSON.
        height: Original image height.
        width: Original image width.

    Returns:
        Binary numpy array of shape (height, width) with dtype bool.
    """
    seg = annotation.get("segmentation")
    if not seg:
        # Fall back to bounding box if segmentation is empty
        bbox = annotation.get("bbox")
        mask = np.zeros((height, width), dtype=bool)
        if bbox and len(bbox) == 4:
            x, y, w, h = map(int, bbox)
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(width, x + w)
            y2 = min(height, y + h)
            mask[y1:y2, x1:x2] = True
        return mask

    if isinstance(seg, list):
        # Polygon segmentation: list of polygons
        return polygon_to_mask(seg, height, width)
    elif isinstance(seg, dict):
        # RLE segmentation
        return rle_to_mask(seg, height, width)
    else:
        logger.warning(f"Unknown segmentation format: {type(seg)}")
        return np.zeros((height, width), dtype=bool)


def build_instance_masks(
    annotations: List[Dict[str, Any]], height: int, width: int
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """Build multi-instance masks and composite instance label map.

    Args:
        annotations: List of instance annotations for an image.
        height: Image height.
        width: Image width.

    Returns:
        Tuple of:
          - instance_masks: np.ndarray of shape (num_instances, height, width) dtype bool
          - composite_labels: np.ndarray of shape (height, width) dtype int32 (0 is background, 1..N instances)
          - parsed_instances: List of metadata dicts for each valid instance
    """
    valid_masks: List[np.ndarray] = []
    parsed_instances: List[Dict[str, Any]] = []
    composite = np.zeros((height, width), dtype=np.int32)

    instance_idx = 1
    for ann in annotations:
        if ann.get("iscrowd", 0) == 1:
            # Note crowd annotations are tracked but separated
            ann_is_crowd = True
        else:
            ann_is_crowd = False

        mask = decode_coco_annotation(ann, height, width)
        if mask.sum() == 0:
            continue

        valid_masks.append(mask)
        parsed_instances.append(
            {
                "instance_id": instance_idx,
                "category_id": ann.get("category_id", -1),
                "bbox": ann.get("bbox", []),
                "area": float(ann.get("area", mask.sum())),
                "iscrowd": ann_is_crowd,
            }
        )
        # Assign pixel value in composite map
        composite[mask] = instance_idx
        instance_idx += 1

    if valid_masks:
        stacked = np.stack(valid_masks, axis=0)
    else:
        stacked = np.zeros((0, height, width), dtype=bool)

    return stacked, composite, parsed_instances


def resize_discrete_mask(mask: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """Resize discrete label or instance mask strictly using nearest-neighbor interpolation.

    MANDATORY RULE: Never use bilinear or bicubic interpolation for categorical or
    instance identity masks, as this fabricates non-existent intermediate class IDs.

    Args:
        mask: 2D array (H, W) or 3D array (N, H, W).
        target_size: (target_height, target_width).

    Returns:
        Resized array with the exact same dtype and unique integer values.
    """
    import cv2

    target_h, target_w = target_size
    orig_dtype = mask.dtype

    if mask.ndim == 2:
        work_mask = mask.astype(np.int32)
        resized = cv2.resize(work_mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        return resized.astype(orig_dtype)
    elif mask.ndim == 3:
        n = mask.shape[0]
        resized_layers = []
        for i in range(n):
            work_mask = mask[i].astype(np.int32)
            res = cv2.resize(work_mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
            resized_layers.append(res.astype(orig_dtype))
        if resized_layers:
            return np.stack(resized_layers, axis=0)
        return np.zeros((0, target_h, target_w), dtype=orig_dtype)
    else:
        raise ValueError(f"Expected 2D or 3D mask array, got shape {mask.shape}")
