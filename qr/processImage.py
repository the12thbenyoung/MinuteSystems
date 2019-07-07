#thanks to:
#https://jdhao.github.io/2019/02/23/crop_rotated_rectangle_opencv/
#https://stackoverflow.com/questions/44926316/how-to-locate-and-read-data-matrix-code-with-python

import matplotlib.pyplot as plt
import numpy as np
import cv2

def pythagoras(p1, p2):
    return sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

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

    # now rotated rectangle becomes vertical and we crop it
    img_crop = cv2.getRectSubPix(img_rot, size, center)

    return img_crop

well = plt.imread('copy.jpg')
well = cv2.cvtColor(well, cv2.COLOR_RGB2GRAY)
plt.subplot(151); plt.title('A')
plt.imshow(well)

harris = cv2.cornerHarris(well,6, 1,0.00)
plt.subplot(152); plt.title('B')
plt.imshow(harris)

x, thr = cv2.threshold(harris, 0.1 * harris.max(), 255, cv2.THRESH_BINARY)
thr = thr.astype('uint8')
plt.subplot(153); plt.title('C')
plt.imshow(thr)

contours, hierarchy = cv2.findContours(thr.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
areas = list(map(lambda x: cv2.contourArea(cv2.convexHull(x)), contours))
max_i = areas.index(max(areas))
d = cv2.drawContours(np.zeros_like(thr), contours, max_i, 255, 1)
plt.subplot(154); plt.title('D')
plt.imshow(d)

rect = cv2.minAreaRect(contours[max_i])

e = crop_rect(well, rect)
cv2.imwrite("rect_crop.jpg", e)

# box = cv2.boxPoints(rect)
# x_dim = np.int0(pythagoras(box[0],box[2]))
# y_dim = np.int0(pythagoras(box[1],box[3]))
# new_image = np.zeros((x_dim, y_dim))

# box = np.int0(box)
# e= cv2.drawContours(well,[box],0,1,1)

# plt.subplot(155); plt.title('E')
# plt.imshow(e)

# plt.show()

