import copy
import os
from typing import List, Optional, Tuple

import cv2
import mmcv
import numpy as np
from matplotlib import pyplot as plt

__all__ = ["visualize_camera", "visualize_map"]

MAP_PALETTE = {
    "drivable_area": (166, 206, 227),
    "road_segment": (31, 120, 180),
    "road_block": (178, 223, 138),
    "lane": (51, 160, 44),
    "ped_crossing": (251, 154, 153),
    "walkway": (227, 26, 28),
    "stop_line": (253, 191, 111),
    "carpark_area": (255, 127, 0),
    "road_divider": (202, 178, 214),
    "lane_divider": (106, 61, 154),
    "divider": (106, 61, 154),
}

def visualize_map(
    fpath: str,
    masks: np.ndarray,
    *,
    classes: List[str],
    background: Tuple[int, int, int] = (240, 240, 240),
) -> None:
    assert masks.dtype == np.bool, masks.dtype

    masks = masks
    canvas = np.zeros((*masks.shape[-2:], 3), dtype=np.uint8)
    canvas[:] = background

    print(f"Masks shape : {canvas.shape}")

    for k, name in enumerate(classes):
        if name in MAP_PALETTE:
            print(name)
            print(masks[k])
            canvas[masks[k], :] = MAP_PALETTE[name]

    print(canvas)
    canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)

    mmcv.mkdir_or_exist(os.path.dirname(fpath))
    mmcv.imwrite(canvas, fpath)