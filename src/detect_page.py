from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from .preprocess import resize_max_dim, preprocess_for_edges
from .io_handling import imwrite, ensure_dir


@dataclass
class DetectionResult:
    corners: Optional[np.ndarray]
    score: float
    method: str
    debug: dict


def order_points(pts: np.ndarray) -> np.ndarray:
    pts = pts.astype(np.float32)
    sorted_by_x = pts[np.argsort(pts[:, 0])]

    left  = sorted_by_x[:2]   # two leftmost points
    right = sorted_by_x[2:]   # two rightmost points

    tl, bl = left[np.argsort(left[:, 1])]
    tr, br = right[np.argsort(right[:, 1])]

    return np.stack([tl, tr, br, bl], axis=0).astype(np.float32)


def quad_area(quad: np.ndarray) -> float:
    # polygon area via contourArea
    return float(cv2.contourArea(quad.reshape(-1, 1, 2).astype(np.float32)))


def is_convex_quad(quad: np.ndarray) -> bool:
    return bool(cv2.isContourConvex(quad.reshape(-1, 1, 2).astype(np.float32)))


def angle_degrees(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ba = a - b
    bc = c - b
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc)) + 1e-9
    cosang = float(np.dot(ba, bc) / denom)
    cosang = max(-1.0, min(1.0, cosang))
    return math.degrees(math.acos(cosang))


def score_quad(quad: np.ndarray, img_shape: tuple[int, int, int]) -> float:
    h, w = img_shape[:2]
    img_area = float(h * w)

    quad = order_points(quad)
    area = quad_area(quad)
    area_ratio = area / (img_area + 1e-9)

    # Area ratio sanity
    if area_ratio < 0.15 or area_ratio > 0.995:
        return 0.0

    # Convexity
    if not is_convex_quad(quad):
        return 0.0

    # Side lengths
    tl, tr, br, bl = quad
    sides = [
        np.linalg.norm(tr - tl),
        np.linalg.norm(br - tr),
        np.linalg.norm(bl - br),
        np.linalg.norm(tl - bl),
    ]
    min_side = min(sides)
    max_side = max(sides)
    if min_side < 0.10 * min(h, w):
        return 0.0
    if max_side / (min_side + 1e-9) > 12.0:
        return 0.0

    # Angles close to 90, but allow perspective
    angles = [
        angle_degrees(bl, tl, tr),
        angle_degrees(tl, tr, br),
        angle_degrees(tr, br, bl),
        angle_degrees(br, bl, tl),
    ]
    # Penalize if any angle is too extreme
    extreme = any(a < 30.0 or a > 150.0 for a in angles)
    if extreme:
        return 0.0

    # Convert area_ratio into a soft score peaked around ~0.6
    # since documents often occupy a big part of the image
    area_score = 1.0 - abs(area_ratio - 0.60) / 0.60
    area_score = max(0.0, min(1.0, area_score))

    # Angle score: closer to 90 is better
    angle_err = sum(abs(a - 90.0) for a in angles) / 4.0
    angle_score = max(0.0, 1.0 - angle_err / 60.0)  # 60 deg average error => 0

    score = 0.65 * area_score + 0.35 * angle_score
    return float(max(0.0, min(1.0, score)))


def find_page_quad_from_contours(edges_closed: np.ndarray, top_k: int = 10) -> tuple[Optional[np.ndarray], str]:
    contours, _ = cv2.findContours(edges_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, "none"

    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:top_k]

    # First try: approxPolyDP to get exactly 4 corners
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            quad = approx.reshape(4, 2).astype(np.float32)
            return quad, "approxPolyDP"

    # Fallback: take largest contour and compute minAreaRect box
    c0 = contours[0]
    rect = cv2.minAreaRect(c0)
    box = cv2.boxPoints(rect)  # (4,2)
    quad = box.astype(np.float32)
    return quad, "minAreaRect"


def draw_debug(
    img_bgr: np.ndarray,
    intermediates: dict,
    quad_resized: Optional[np.ndarray],
    scale: float,
    method: str,
    score: float,
) -> dict: # holds debug images (BGR or gray)
    debug = {}

    debug["gray"] = intermediates["gray"]
    debug["edges"] = intermediates["edges"]
    debug["edges_closed"] = intermediates["edges_closed"]

    # Overlay on resized image
    overlay = img_bgr.copy()
    if quad_resized is not None:
        quad_ord = order_points(quad_resized)
        # Draw polygon
        pts = quad_ord.reshape(-1, 1, 2).astype(np.int32)
        cv2.polylines(overlay, [pts], True, (0, 255, 0), 2)

        # Draw corners + labels
        labels = ["TL", "TR", "BR", "BL"]
        for (x, y), lab in zip(quad_ord, labels):
            cv2.circle(overlay, (int(x), int(y)), 6, (0, 0, 255), -1)
            cv2.putText(
                overlay,
                lab,
                (int(x) + 8, int(y) - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

    cv2.putText(
        overlay,
        f"method={method} score={score:.2f} scale={scale:.3f}",
        (12, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 0, 0),
        2,
        cv2.LINE_AA,
    )
    debug["overlay_resized"] = overlay

    return debug


def save_debug_images(debug_dir: str, debug: dict) -> None:
    ensure_dir(debug_dir)
    for name, img in debug.items():
        path = f"{debug_dir}/{name}.png"
        # If grayscale, write directly: also fine if BGR
        imwrite(path, img)

# Returns corners in original image coordinates
def detect_page_corners(
    img_bgr_original: np.ndarray,
    debug_dir: Optional[str] = None,
    max_dim: int = 1200,
    use_clahe: bool = False,
) -> DetectionResult:
    img_resized, scale = resize_max_dim(img_bgr_original, max_dim=max_dim)

    # Preprocess resized
    inter = preprocess_for_edges(
        img_resized,
        use_clahe=use_clahe,
        blur_ksize=5,
        canny_lo=60,
        canny_hi=180,
        close_ksize=7,
    )

    quad_resized, method = find_page_quad_from_contours(inter["edges_closed"], top_k=10)

    if quad_resized is None:
        score = 0.0
        corners_original = None
        dbg = draw_debug(img_resized, inter, None, scale, "none", score)
        if debug_dir:
            save_debug_images(debug_dir, dbg)
        return DetectionResult(corners=corners_original, score=score, method="none", debug=dbg)

    quad_resized = order_points(quad_resized)
    score = score_quad(quad_resized, img_resized.shape)

    # If score is too low, treat as failure (but still output debug)
    if score < 0.20:
        corners_original = None
        method_used = method + "_rejected"
    else:
        # Map corners back to original coordinates:
        # resized = original * scale  => original = resized / scale
        corners_original = (quad_resized / scale).astype(np.float32)
        method_used = method

    dbg = draw_debug(img_resized, inter, quad_resized, scale, method_used, score)
    if debug_dir:
        save_debug_images(debug_dir, dbg)

    return DetectionResult(corners=corners_original, score=score, method=method_used, debug=dbg)
