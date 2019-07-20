import matplotlib.pyplot as plt
import numpy as np
import cv2
from findTubes import process_matrix, find_largest_contour, crop_smallest_rect

from pylibdmtx.pylibdmtx import decode

img = cv2.imread('badtest.png')
print(decode(img))

