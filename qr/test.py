import matplotlib.pyplot as plt
import numpy as np
import cv2
from findTubes import process_matrix, find_largest_contour, crop_smallest_rect

from pylibdmtx.pylibdmtx import decode

def show_image_small(name, img):
    cv2.imshow(name, cv2.resize(img, (int(img.shape[1]/5), int(img.shape[0]/5)), interpolation=cv2.INTER_AREA))
    cv2.waitKey(0)

#for thresholding matrix after it's been cropped from tube
MATRIX_THRESH_FACTOR = 0.3

img = cv2.imread('images/badkek34.jpg')

while MATRIX_THRESH_FACTOR < 0.45:
    x, tube_thr = cv2.threshold(img, MATRIX_THRESH_FACTOR * img.max(), 255, cv2.THRESH_BINARY)
    print(decode(tube_thr))
    MATRIX_THRESH_FACTOR += 0.01

cv2.imshow('thr', tube_thr)
cv2.waitKey(0)


