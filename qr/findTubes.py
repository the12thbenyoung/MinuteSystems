#thanks to:
#boundingRec://gist.github.com/bigsnarfdude/d811e31ee17495f82f10db12651ae82d#file-gistfile1-txt-L33k

import cv2
import numpy as np
from processImage import processMatrix

#max number of pixels of different between y coordinates for tubes to be considered in same row
SAME_ROW_THRESHOLD = 70

NUM_ROWS = 8
NUM_COLS = 12

RACK_THRESH_FACTOR = 0.7
TUBES_THRESH_FACTOR = 0.5

#should be: 80

def show_image(name, img):
    cv2.imshow(name, cv2.resize(img, (int(img.shape[1]/5), int(img.shape[0]/5)), interpolation=cv2.INTER_AREA))

def crop_rect(img, threshFactor):
    x, thr = cv2.threshold(img, threshFactor * img.max(), 255, cv2.THRESH_BINARY)

    #get outer contour of data matrix
    contours, _ = cv2.findContours(thr.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    areas = list(map(lambda x: cv2.contourArea(cv2.convexHull(x)), contours))
    max_i = areas.index(max(areas))

    rect = cv2.minAreaRect(contours[max_i])

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

def crop_data_matrix(img, blockSize, threshFactor):
    #detect corners
    harris = cv2.cornerHarris(well, blockSize, 1, 0.00)

    # harris = harris.astype('uint8')
    return crop_rect(harris, threshFactor)

if __name__ == '__main__':
    img = cv2.imread('tubesmissing.jpg')

    #convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #crop to just rack, threshold and to get tubes in white
    rack = crop_rect(gray, RACK_THRESH_FACTOR)
    show_image('rack', rack)
    x, rack_thr = cv2.threshold(rack, TUBES_THRESH_FACTOR * rack.max(), 255, cv2.THRESH_BINARY)
    show_image('rack_thr', rack_thr)
    cv2.waitKey(0)
    rack_thr = 255 - rack_thr


    #get contours around tubes
    contours, hierarchy = cv2.findContours(rack.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #each set holds a group of points with similar y-coordinate - corresponding to one row in the rack
    rowLists = []
    #dict to associate (x,y) coordinate pairs with decoded outputs.
    #get hash value with 100*x + y
    coorToData = {}
    #tuples of (indices, decoded data)
    dataIndices = []

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
                coorToData[hash((x,y))] = data[0].data

                if x > maxX:
                    maxX = x
                elif x < minX:
                    minX = x

                newRow = True
                #add (x,y) tuple from rect to its row, depending on its y-coordinate
                for row in rowLists:
                    #put tube in row if its y coordinate is close enough to that of first member
                    if abs(y - row[0][1]) < SAME_ROW_THRESHOLD:
                        row.append((x,y))
                        newRow = False
                #put this in its own row if we haven't found one it fits in
                if newRow and len(rowLists) < NUM_ROWS:
                   rowLists.append([(x,y)])

                i += 1

    #approx horizontal distance between each tube
    horizDist = (maxX - minX)/(NUM_COLS-1)

    #sort by y-pixel of first element to find order of rows
    rowLists.sort(key = (lambda row: row[0][1]))

    for row in rowLists:
        print(row)

    #for each found row, determine x index by distance from minX
    for row in range(len(rowLists)):
        y_index = row + 1
        for tube in rowLists[row]:
            x_index = int(np.round((tube[0] - minX)/horizDist)) + 1
            dataIndices.append(((x_index, y_index), coorToData[hash((tube[0], tube[1]))]))

    print(dataIndices)

    print(i)

