#thanks to:
#boundingRec://gist.github.com/bigsnarfdude/d811e31ee17495f82f10db12651ae82d#file-gistfile1-txt-L33k

import cv2
import numpy as np
from processImage import processMatrix

#max number of pixels of different between y coordinates for tubes to be considered in same row
SAME_ROW_THRESHOLD = 30

img = cv2.imread('qrtestphone2.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

x, thr = cv2.threshold(gray, 0.4 * gray.max(), 255, cv2.THRESH_BINARY)

#invert image
thr = 255 - thr

# cv2.imshow('thresh', thr)

contours, hierarchy = cv2.findContours(thr.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

img_copy = img.copy()

#each set holds a group of points with similar y-coordinate - corresponding to one row in the rack
rowLists = [[] for i in range(8)]
#keep track of how many lists in rowLists have at least one member
rowsFound = 0
#dict to associate (x,y) coordinate pairs with decoded outputs.
#get hash value with 100*x + y
coorToData = {}

i = 0
#smallest and largest x-coordinate found
maxX = -1*float('inf')
minX = float('inf')
for contour in contours:
    #smallest non-rotated bounding rectangle
    rect = cv2.boundingRect(contour)
    x, y, w, h = rect

    #hopefully only the tube contours are this square and this big
    if 80 < w and 160 > w and 80 < h and 160 > h:
        img_crop= img[y:y+h, x:x+w]

        data = processMatrix(img_crop)
        if data:    
            #add decoded data to coordinate-data dict
            coorToData[hash((x,y))] = data.data

            if x > maxX:
                maxX = x
            elif x < minX:
                minX = x

            newRow = True
            #add (x,y) tuple from rect to its row, depending on its y-coordinate
            for row in range(rowsFound):
                #put tube in row if its y coordinate is close enough to that of first member
                if abs(y - rowLists[row][1]) < SAME_ROW_THRESHOLD:
                    rowLists[row].append((x,y))
                    newRow = False
            #put this in its own row if we haven't found one it fits in
            if newRow and rowsFound < 8:
               rowLists[rowsFound].append((x,y))
               rowsFound += 1

            i = += 1

#approx horizontal distance between each tube
horizDist = (maxX - minX)/7

#for each found row, determine x index by distance from minX


print(i)
cv2.imshow("contours", img_copy)
# cv2.imwrite('contours.jpg', img_copy)

cv2.waitKey(0)
cv2.destroyAllWindows()

#for drawing numbers on image
def annotateImage(img, data):
    cv2.putText(img, str(data[0].data), (x,y+70), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), thickness=2)
    cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
