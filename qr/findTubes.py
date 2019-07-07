#thanks to:
#boundingRec://gist.github.com/bigsnarfdude/d811e31ee17495f82f10db12651ae82d#file-gistfile1-txt-L33k

import cv2
import numpy as np
from processImage import processMatrix

img = cv2.imread('qrtestphone2.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

x, thr = cv2.threshold(gray, 0.4 * gray.max(), 255, cv2.THRESH_BINARY)

#invert image
thr = 255 - thr

# cv2.imshow('thresh', thr)

contours, hierarchy = cv2.findContours(thr.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

i = 0
for contour in contours:
    #smallest non-rotated bounding rectangle
    rect = cv2.boundingRect(contour)
    x, y, w, h = rect

    #hopefully only the tube contours are this square and this big
    if 80 < w and 160 > w and 80 < h and 160 > h:
        # cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
        img_crop= img[y:y+h, x:x+w]

        data = processMatrix(img_crop)
        print(data)
        if data:
            i = i + 1
        # cv2.imwrite('images/testcrop{}.jpg'.format(i), img_crop)

print(i)

# cv2.imshow("contours", img)

# cv2.waitKey(0)
# cv2.destroyAllWindows()
