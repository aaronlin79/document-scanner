from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from .detect_page import order_points, DetectionResult


def four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    # Perspective transform from the given corners
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))

    dst = np.array([
        [0,            0            ],
        [maxWidth - 1, 0            ],
        [maxWidth - 1, maxHeight - 1],
        [0,            maxHeight - 1],
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped


def warp_from_result(img_bgr: np.ndarray, result: DetectionResult) -> Optional[np.ndarray]:
    # Deskewed document image, or None if no corners were detected.
    if result.corners is None:
        return None
    return four_point_transform(img_bgr, result.corners)
