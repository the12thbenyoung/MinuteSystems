import matplotlib.pyplot as plt
import numpy as np
import cv2

from pylibdmtx.pylibdmtx import decode
from os import listdir
from multiprocessing import Pool, Process, Queue

from dataMatrixDecoder import process_rack

img = cv2.imread('images/badkek8.jpg')
grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
rack_blur = cv2.GaussianBlur(grey,(5,5),0)
thr = cv2.adaptiveThreshold(rack_blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                                   cv2.THRESH_BINARY_INV,301,0)

cv2.imshow('thr', thr)
cv2.waitKey(0)

contours, hierarchy = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(len(contours))

