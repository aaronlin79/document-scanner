# Document Scanner

A simple document scanner built with well-known computer vision methods in Python and OpenCV.

This project takes a photo of a document, detects the page boundary, corrects perspective distortion, and produces a high-contrast scanned result. It was developed as part of a computer vision course (EE146 @ UCR) and focuses on implementing the full pipeline from edge detection to perspective rectification and document enhancement.


## Features

- Detects the document boundary from an image using contour analysis
- Scores candidate quadrilaterals based on area, convexity, side lengths, and corner angles
- Applies a four-point perspective transform to deskew the document
- Produces a binarized scan-like image using adaptive thresholding
- Supports optional CLAHE preprocessing for difficult lighting conditions
- Saves intermediate debugging artifacts such as grayscale, edges, closed edges, and detection overlays
- Includes both single-image testing and batch evaluation scripts


## Usage

### Single Image Scan:
```
python -m scripts.single_img_test --img <image-path> --outdir <output-dir>
```

Example:
```
python -m scripts.single_img_test --img documents/raw/base.jpg --outdir outputs/debug/single_img
```

### Batch Processing:
```
python -m scripts.multi_img_test --input <input-dir> --outdir <output-dir>
```

Example:
```
python -m scripts.multi_img_test --input documents/raw  --outdir outputs/debug/eval
```
