import cv2
import numpy as np


def thresh_document(img):

    # Thresholding
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Reduce noise by using Gaussian BLur
    blur_img = cv2.GaussianBlur(gray_img, (5,5), 0)

    thresh_img = cv2.adaptiveThreshold(blur_img, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 10)

    # Opening to clean small noise
    kernel = np.ones((3,3), np.uint8)
    opened_img = cv2.morphologyEx(thresh_img, cv2.MORPH_OPEN, kernel)


    return opened_img
