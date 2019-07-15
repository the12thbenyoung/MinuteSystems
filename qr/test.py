import matplotlib.pyplot as plt
import numpy as np
import cv2

from pylibdmtx.pylibdmtx import decode

img = cv2.imread('images/tube25.jpg')

# cv2.imshow('thr', thr)
# cv2.waitKey(0)

for i in range(100):
    data = decode(img)

