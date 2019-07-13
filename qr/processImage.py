#thanks to:
#https://jdhao.github.io/2019/02/23/crop_rotated_rectangle_opencv/
#https://stackoverflow.com/questions/44926316/how-to-locate-and-read-data-matrix-code-with-python

import numpy as np
import cv2
from pylibdmtx.pylibdmtx import decode

def crop_rect(img, rect):
    # get the parameter of the small rectangle
    center, size, angle = rect[0], rect[1], rect[2]
    center, size = tuple(map(int, center)), tuple(map(int, size))

    # get row and col num in img
    height, width = img.shape[0], img.shape[1]

    # calculate the rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1)
    # rotate the original image
    img_rot = cv2.warpAffine(img, M, (width, height))

    x,y = size
    size = (x+4, y+4)

    # now rotated rectangle becomes vertical and we crop it
    img_crop = cv2.getRectSubPix(img_rot, size, center)

    return img_crop

def processMatrix(img):
    # well = plt.imread('images/testcrop9.jpg')
    # # well = plt.imread('copy.jpg')


    #binary threshold
    x, thr = cv2.threshold(harris, 0.01 * harris.max(), 255, cv2.THRESH_BINARY)
    thr = thr.astype('uint8')

    #get outer contour of data matrix
    contours, hierarchy = cv2.findContours(thr.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    areas = list(map(lambda x: cv2.contourArea(cv2.convexHull(x)), contours))
    max_i = areas.index(max(areas))
    # d = cv2.drawContours(np.zeros_like(thr), contours, max_i, 255, 1)

    matrix = crop_rect(well, cv2.minAreaRect(contours[max_i]))

    data = decode(matrix)
    return data

#for drawing numbers on image
def annotateImage(img, data):
    cv2.putText(img, str(data[0].data), (x,y+70), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), thickness=2)
    cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
