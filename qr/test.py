import matplotlib.pyplot as plt
import numpy as np
import cv2

from pylibdmtx.pylibdmtx import decode

img = cv2.imread('images/tube20.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imshow('before', img)

neg = 255 - img
x, thr = cv2.threshold(neg, 0.82 * neg.max(), 255, cv2.THRESH_BINARY)

thr = thr.astype('uint8')

#get outer contour of data matrix
contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

areas = list(map(lambda x: cv2.contourArea(cv2.convexHull(x)), contours))
max_i = np.argmax(areas)


rect = cv2.boundingRect(contours[max_i])
x, y, w, h = rect

tube = img[y+5:y+h-5, x+5:x+w-5]

cv2.imshow('after', tube)
cv2.waitKey(0)
