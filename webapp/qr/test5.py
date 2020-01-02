import numpy as np
import cv2

from pylibdmtx.pylibdmtx import decode
from os import listdir

img = cv2.imread('images/badkek8.jpg')
grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
rack_blur = cv2.GaussianBlur(grey,(5,5),0)
thr = cv2.adaptiveThreshold(rack_blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                                   cv2.THRESH_BINARY_INV,301,5)

cv2.imshow('thr', thr)
cv2.waitKey(0)

print(decode(thr)[0].data)
