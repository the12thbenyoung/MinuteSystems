import cv2
import numpy as np

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


img = cv2.imread('tubesmissing.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("gray", cv2.resize(gray, (int(gray.shape[1]/5), int(gray.shape[0]/5)), interpolation=cv2.INTER_AREA))
cv2.waitKey(0)

x, thr = cv2.threshold(gray, 0.7 * gray.max(), 255, cv2.THRESH_BINARY)
cv2.imshow("thr", cv2.resize(thr, (int(thr.shape[1]/5), int(thr.shape[0]/5)), interpolation=cv2.INTER_AREA))
cv2.waitKey(0)

#get outer contour of data matrix
contours, hierarchy = cv2.findContours(thr.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
print(len(contours))
areas = list(map(lambda x: cv2.contourArea(cv2.convexHull(x)), contours))
max_i = areas.index(max(areas))
d = cv2.drawContours(np.zeros_like(thr), contours, max_i, 255, 1)
cv2.imshow("d", cv2.resize(d, (int(d.shape[1]/5), int(d.shape[0]/5)), interpolation=cv2.INTER_AREA))
cv2.waitKey(0)

matrix = crop_rect(img, cv2.minAreaRect(contours[max_i]))
cv2.imshow("matrix", cv2.resize(matrix, (int(matrix.shape[1]/5), int(matrix.shape[0]/5)), interpolation=cv2.INTER_AREA))
cv2.waitKey(0)

#get outer contour of data matrix
contours, hierarchy = cv2.findContours(thr.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

