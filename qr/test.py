import matplotlib.pyplot as plt
import numpy as np
import cv2

# well = plt.imread('images/tube19.jpg')
well = cv2.imread('images/tube19.jpg')
well = cv2.cvtColor(well, cv2.COLOR_BGR2GRAY)
print(well)
plt.subplot(151); plt.title('A')
plt.imshow(well)

harris = cv2.cornerHarris(well,8,1,0.00)
cv2.imshow('harris', harris)
cv2.waitKey(0)

plt.subplot(152); plt.title('B')
plt.imshow(harris)

x, thr = cv2.threshold(harris, 0.01 * harris.max(), 255, cv2.THRESH_BINARY)
thr = thr.astype('uint8')
plt.subplot(153); plt.title('C')
plt.imshow(thr)

contours, hierarchy = cv2.findContours(thr.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
areas = list(map(lambda x: cv2.contourArea(cv2.convexHull(x)), contours))
max_i = np.argmax(areas)
d = cv2.drawContours(np.zeros_like(thr), contours, max_i, 255, 1)
plt.subplot(154); plt.title('D')
plt.imshow(d)

rect =cv2.minAreaRect(contours[max_i])
box = cv2.boxPoints(rect)
box = np.int0(box)
e= cv2.drawContours(well,[box],0,1,1)
plt.subplot(155); plt.title('E')
plt.imshow(e)

plt.show()
