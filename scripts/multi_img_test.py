import argparse
import csv
from pathlib import Path
import cv2

from src.io_handling import imread_color, ensure_dir
from src.detect_page import detect_page_corners
from src.threshholding import thresh_document
from src.transform import warp_from_result


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def iter_images(folder: str):
    p = Path(folder)
    for fp in sorted(p.rglob("*")):
        if fp.suffix.lower() in IMG_EXTS:
            yield fp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Folder of images")
    ap.add_argument("--outdir", default="outputs/debug/eval", help="Where to write logs/debug")
    ap.add_argument("--max-dim", type=int, default=1200)
    ap.add_argument("--clahe", action="store_true")
    ap.add_argument("--save-failures", action="store_true", help="Save debug images only for failures")
    args = ap.parse_args()

    ensure_dir(args.outdir)
    out_csv = Path(args.outdir) / "detection_log.csv"

    total = 0
    ok = 0
    scores = []

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "method", "score", "success"])

        for fp in iter_images(args.input):
            total += 1
            img = imread_color(str(fp))

            debug_dir = None
            # If save-failures: only save debug if it fails
            if not args.save_failures:
                debug_dir = str(Path(args.outdir) / "debug_all" / fp.stem)

            result = detect_page_corners(img, debug_dir=debug_dir, max_dim=args.max_dim, use_clahe=args.clahe)

            success = result.corners is not None
            if success:
                ok += 1
                scores.append(result.score)

                warped_img = warp_from_result(img, result)
                thresh_img = thresh_document(warped_img)
                
                out_scan_path = Path(args.outdir) / "Multi_scan" / f"{fp.stem}_scanned.png"
                out_scan_path.parent.mkdir(parents=True, exist_ok=True)
                
                cv2.imwrite(str(out_scan_path), thresh_img)

            else:
                if args.save_failures:
                    debug_dir = str(Path(args.outdir) / "failures" / fp.stem)
                    # rerun to save debug
                    _ = detect_page_corners(img, debug_dir=debug_dir, max_dim=args.max_dim, use_clahe=args.clahe)

            writer.writerow([str(fp), result.method, f"{result.score:.4f}", int(success)])

    avg_score = sum(scores) / len(scores) if scores else 0.0
    print(f"Images: {total}")
    print(f"Success: {ok} ({(ok/total*100.0 if total else 0.0):.1f}%)")
    print(f"Avg score on successes: {avg_score:.3f}")
    print(f"Log written to: {out_csv}")


if __name__ == "__main__":
    main()