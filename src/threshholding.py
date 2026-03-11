import cv2
import numpy as np


def thresh_document(img):

    # Thresholding
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    thresh_img = cv2.adaptiveThreshold(gray_img, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10)

    # Opening to clean small noise
    kernel = np.ones((3,3), np.uint8)
    opened_img = cv2.morphologyEx(thresh_img, cv2.MORPH_OPEN, kernel)
    closed_img = cv2.morphologyEx(opened_img, cv2.MORPH_CLOSE, kernel)   


    return closed_img
