import cv2
import numpy as np

# Resize image so that max(height, width) == max_dim (if larger)
# Returns resized image and scale where:
#   resized = original * scale
# original = resized / scale
def resize_max_dim(img: np.ndarray, max_dim: int = 1200) -> tuple[np.ndarray, float]:
    h, w = img.shape[:2]
    m = max(h, w)
    if m <= max_dim:
        return img.copy(), 1.0

    scale = max_dim / float(m)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale


def preprocess_for_edges(
    img_bgr: np.ndarray,
    use_clahe: bool = False,
    blur_ksize: int = 5,
    canny_lo: int = 60,
    canny_hi: int = 180,
    close_ksize: int = 7,
) -> dict:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray2 = clahe.apply(gray)
    else:
        gray2 = gray

    k = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
    blurred = cv2.GaussianBlur(gray2, (k, k), 0)

    edges = cv2.Canny(blurred, canny_lo, canny_hi)

    # Morph close to connect broken edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (close_ksize, close_ksize))
    edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Returns dict of intermediate files
    return {
        "gray": gray,
        "gray_eq": gray2,
        "blurred": blurred,
        "edges": edges,
        "edges_closed": edges_closed,
    }