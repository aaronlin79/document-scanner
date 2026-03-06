import argparse
from pathlib import Path

import cv2

from src.io_handling import imread_color, ensure_dir
from src.detect_page import detect_page_corners


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", required=True, help="Path to input image")
    ap.add_argument("--outdir", default="outputs/debug/run_one", help="Debug output directory")
    ap.add_argument("--max-dim", type=int, default=1200, help="Max dimension for detection resize")
    ap.add_argument("--clahe", action="store_true", help="Use CLAHE (sometimes helps with shadows)")
    args = ap.parse_args()

    img = imread_color(args.img)
    ensure_dir(args.outdir)

    result = detect_page_corners(img, debug_dir=args.outdir, max_dim=args.max_dim, use_clahe=args.clahe)

    print(f"method: {result.method}")
    print(f"score:  {result.score:.3f}")
    if result.corners is None:
        print("corners: None (detection failed or rejected)")
        return

    print("corners (original coords):")
    for i, (x, y) in enumerate(result.corners):
        print(f"  {i}: ({x:.1f}, {y:.1f})")

    # Quick visualization on original image
    vis = img.copy()
    corners = result.corners
    # draw polygon
    pts = corners.reshape(-1, 1, 2).astype("int32")
    cv2.polylines(vis, [pts], True, (0, 255, 0), 3)
    for (x, y) in corners:
        cv2.circle(vis, (int(x), int(y)), 8, (0, 0, 255), -1)

    out_vis = str(Path(args.outdir) / "overlay_original.png")
    cv2.imwrite(out_vis, vis)
    print(f"Saved: {out_vis}")


if __name__ == "__main__":
    main()