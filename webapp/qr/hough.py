import cv2
import numpy as np

img = cv2.imread('test.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
_, thr = cv2.threshold(gray, 0.4 * gray.max(), 255, cv2.THRESH_BINARY)

KERNEL_SIZE = 3
kernel = np.ones((KERNEL_SIZE, KERNEL_SIZE), np.uint8)

opening = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel)

harris = cv2.cornerHarris(opening, 6, 1, 0.00)

_, thr2 = cv2.threshold(harris, 0.005 * harris.max(), 255, cv2.THRESH_BINARY)

dilation = cv2.dilate(thr2, kernel, iterations=1)

cv2.imshow('opening', opening)
cv2.imshow('dilation', dilation)
cv2.waitKey(0)

# lines = cv2.HoughLines(edges,1,np.pi/180,100)

# numLines = 0
# for line in lines:
#     rho,theta = line[0]
#     numLines += 1
#     a = np.cos(theta)
#     b = np.sin(theta)
#     x0 = a*rho
#     y0 = b*rho
#     x1 = int(x0 + 1000*(-b))
#     y1 = int(y0 + 1000*(a))
#     x2 = int(x0 - 1000*(-b))
#     y2 = int(y0 - 1000*(a))

#     cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)

# print(numLines)

# cv2.imshow('lines', img)
# cv2.waitKey(0)
