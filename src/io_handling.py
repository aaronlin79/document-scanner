import os
from pathlib import Path
import cv2


def ensure_dir(path: str | os.PathLike) -> str:
    Path(path).mkdir(parents=True, exist_ok=True)
    return str(path)


def imread_color(path: str) -> "cv2.Mat":
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def imwrite(path: str, img: "cv2.Mat") -> None:
    ensure_dir(Path(path).parent)
    ok = cv2.imwrite(path, img)
    if not ok:
        raise IOError(f"Could not write image: {path}")