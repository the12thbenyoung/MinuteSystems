import matplotlib.pyplot as plt
import numpy as np
import cv2
from findTubes import process_matrix, find_largest_contour

from pylibdmtx.pylibdmtx import decode

# for i in range(93):
#     img = cv2.imread(f'images/tube{i}.jpg')
#     _, rack_thr = cv2.threshold(img, 0.4* img.max(), 255, cv2.THRESH_BINARY)
#     cv2.imshow('rack_thr', rack_thr)
#     cv2.waitKey(0)

img = cv2.imread('thresh_test.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, thr = cv2.threshold(gray, 0.4* gray.max(), 255, cv2.THRESH_BINARY)

#detect corners
harris = cv2.cornerHarris(thr, 6, 1, 0.00)

contour = find_largest_contour(harris, threshFactor)
contourArea = cv2.contourArea(cv2.convexHull(contour)),

#if area is too small, threshFactor was too high (didn't find whole matrix),
#so lower it and try again
while contourArea[0] < MIN_MATRIX_CONTOUR_AREA and threshFactor > 0:
    threshFactor -= 0.002
    contour = find_largest_contour(harris, threshFactor)
    contourArea = cv2.contourArea(cv2.convexHull(contour)),

#crop out matrix from tube
matrix = crop_smallest_rect(img, contour)
cv2.imshow('matrix', matrix)
cv2.waitKey(0)

#keep raising threshold until matrix becomes sufficiently small, meaning
#it's been cropped to just the matrix and not the surrounding numbers
while any([dim > MATRIX_SIZE_UPPER_BOUND
          for dim in [matrix.shape[0], matrix.shape[1]]]) and threshFactor < 1:
    threshFactor += 0.002
    # print(threshFactor)
    contour = find_largest_contour(harris, threshFactor)
    contourArea = cv2.contourArea(cv2.convexHull(contour)),
    matrix = crop_smallest_rect(img, contour)

#empty slots have weird aspect ratios (tall or wide) after they're processed. We 
#can use this to filter out some of them
height, width = matrix.shape[0], matrix.shape[1]
if height/width < 1.5 and width/height < 1.5:
    data = decode(matrix)
else:
    data = None

print(data)


