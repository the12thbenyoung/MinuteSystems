#thanks to:
#boundingRec://gist.github.com/bigsnarfdude/d811e31ee17495f82f10db12651ae82d#file-gistfile1-txt-L33k

import cv2
import numpy as np

img = cv2.imread('qrtestphone2.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

x, thr = cv2.threshold(gray, 0.4 * gray.max(), 255, cv2.THRESH_BINARY)

#invert image
thr = 255 - thr

# cv2.imshow('thresh', thr)

contours, hierarchy = cv2.findContours(thr.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

i = 0
for c in contours:
    if i == 0:
        rect = cv2.boundingRect(c)
        x, y, w, h = rect
        if 80 < w and 150 > w and 80 < h and 150 > h:
            print(rect)
            # cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
            img_crop= img[y:y+h, x:x+w]
            cv2.imwrite('testcrop.jpg', img_crop)
            i = i + 1

print(i)

# cv2.imshow("contours", img)

# areas = list(map(lambda x: cv2.contourArea(cv2.convexHull(x)), contours))
# max_i = areas.index(max(areas))
# d = cv2.drawContours(np.zeros_like(thr), contours, max_i, 255, 1)
# cv2.imshow('test', d)


cv2.waitKey(0)
cv2.destroyAllWindows()
