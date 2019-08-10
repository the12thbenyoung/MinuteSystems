import numpy as np
import cv2
from matplotlib import pyplot as plt
from findTubes import *

img = cv2.imread('chungis.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

rack_blur = cv2.GaussianBlur(gray,(5,5),0)
thresh = cv2.adaptiveThreshold(rack_blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                                   cv2.THRESH_BINARY_INV,301,TUBE_AREA_THRESH_CONSTANT)

#noise removal
kernel = np.ones((3,3),np.uint8)
sure_bg = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 5)
# sure background area
# sure_bg = cv2.dilate(opening,kernel,iterations=3)
# # Finding sure foreground area
# dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
# ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
# # Finding unknown region
# sure_fg = np.uint8(sure_fg)
# unknown = cv2.subtract(sure_bg,sure_fg)

# show_image_small(['sure_bg', sure_bg], ['sure_fg', sure_fg])
show_image_small(['sure_bg', sure_bg])
