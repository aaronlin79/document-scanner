# Document Scanner

A document scanner built with well-known computer vision methods in Python and OpenCV.

This project takes a photo of a document, detects the page boundary, corrects perspective distortion, and produces a high-contrast scanned result. It was developed as part of a computer vision course (EE146 @ UCR) and focuses on implementing the full pipeline from edge detection to perspective rectification and document enhancement.


## Features

- Detects the document boundary from an image using contour analysis
- Scores candidate quadrilaterals based on area, convexity, side lengths, and corner angles
- Applies a four-point perspective transform to deskew the document
- Produces a binarized scan-like image using adaptive thresholding
- Supports optional CLAHE preprocessing for difficult lighting conditions
- Saves intermediate debugging artifacts such as grayscale, edges, closed edges, and detection overlays
- Includes both single-image testing and batch evaluation scripts


## Dataset

This project uses the **GNHK (GoodNotes Handwriting Kollection)** dataset for evaluation and benchmarking of document boundary detection and perspective correction.

The GNHK dataset consists of unconstrained, camera-captured images of English handwritten documents collected from diverse real-world environments. It is designed to reflect realistic variations in lighting, orientation, background clutter, and handwriting styles, making it well-suited for evaluating document analysis and computer vision pipelines.

The dataset was introduced in:

> Lee, A. W. C., Chung, J., & Lee, M. (2021). *GNHK: A Dataset for English Handwriting in the Wild*. ICDAR 2021.

- Official page: https://www.goodnotes.com/gnhk/
- GitHub repository: https://github.com/GoodNotes/GNHK-dataset


## Usage

### Uploading Images:
Place your image anywhere in the project (recommended: `documents/raw/`)

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


## Future Work
- Deep Learning Document Detection (e.g. YOLO, U-Nets)
- Improved Corner Detection (e.g. learning-based keypoint detection, sub-pixel corner refinement)
- Mobile/Real-Time Deployment
- Intuitive User Interface


## Acknowledgements

This project was developed as part of **EE146 (Computer Vision)** at the University of California, Riverside, under the instruction of Professor Bir Bhanu, with guidance from Ankith Jain Rakesh Kumar.
