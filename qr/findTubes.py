#thanks to:
#boundingRec://gist.github.com/bigsnarfdude/d811e31ee17495f82f10db12651ae82d#file-gistfile1-txt-L33k

import cv2
import numpy as np
from pylibdmtx.pylibdmtx import decode

SHOW_IMAGES = False

#max number of pixels of different between y coordinates for tubes to be considered in same row
SAME_ROW_THRESHOLD = 70

NUM_ROWS = 8
NUM_COLS = 12

RACK_THRESH_FACTOR = 0.75
TUBES_THRESH_FACTOR = 0.45
HARRIS_THRESH_FACTOR = 0.01

HARRIS_BLOCK_SIZE = 6

EDGE_LOWER_BOUND = 90
EDGE_UPPER_BOUND = 190

#should be: 80

def show_image_small(name, img):
    cv2.imshow(name, cv2.resize(img, (int(img.shape[1]/5), int(img.shape[0]/5)), interpolation=cv2.INTER_AREA))
    cv2.waitKey(0)

def crop_rect(img, mask, threshFactor):
    x, thr = cv2.threshold(mask, threshFactor * mask.max(), 255, cv2.THRESH_BINARY)
    thr = thr.astype('uint8')

    if SHOW_IMAGES:
        cv2.imshow('thr', thr)

    #get outer contour of data matrix
    contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    areas = list(map(lambda x: cv2.contourArea(cv2.convexHull(x)), contours))
    max_i = np.argmax(areas)

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

    if SHOW_IMAGES:
        cv2.imshow('crop', img_crop)
        cv2.waitKey(0)

    return img_crop

def process_matrix(img, blockSize, threshFactor, i):
    #detect corners
    harris = cv2.cornerHarris(img, HARRIS_BLOCK_SIZE, 1, 0.00)

    #crop out matrix from tube
    matrix = crop_rect(img, harris, threshFactor)

    # cv2.imwrite('images/tube{}.jpg'.format(i), matrix)

    #empty slots have weird aspect ratios (tall or wide) after they're processed. We 
    #can use this to filter out some of them
    height, width = matrix.shape[0], matrix.shape[1]
    if height/width < 1.5 and width/height < 1.5:
        return decode(matrix)
    else:
        # print('zoinks')
        return None

if __name__ == '__main__':
    img = cv2.imread('tubesmissing.jpg')

    #convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #crop to just rack, threshold and invert to get tubes in white
    rack = crop_rect(gray, gray, RACK_THRESH_FACTOR)
    x, rack_thr = cv2.threshold(rack, TUBES_THRESH_FACTOR * rack.max(), 255, cv2.THRESH_BINARY)
    rack_thr = 255 - rack_thr

    # show_image_small('rack_thr', rack_thr)
    # exit(0)

    #get contours around tubes
    contours, hierarchy = cv2.findContours(rack_thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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
        if all([EDGE_LOWER_BOUND < dim and EDGE_UPPER_BOUND > dim for dim in [w,h]]):
            tube = rack[y:y+h, x:x+w]

            if SHOW_IMAGES:
                cv2.imshow('tube', tube)

            data = process_matrix(tube, HARRIS_BLOCK_SIZE, HARRIS_THRESH_FACTOR, i)

            if data:
                i += 1

            #add decoded data to coordinate-data dict
            coorToData[hash((x,y))] = data[0].data if data else 0

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
            if newRow: #and len(rowLists) < NUM_ROWS:
               rowLists.append([(x,y)])


    #approx horizontal distance between each tube
    horizDist = (maxX - minX)/(NUM_COLS-1)

    #sort by y-pixel of first element to find order of rows
    rowLists.sort(key = (lambda row: row[0][1]))

    #for each found row, determine x index by distance from minX
    for row in range(len(rowLists)):
        y_index = row + 1
        for tube in rowLists[row]:
            x_index = int(np.round((tube[0] - minX)/horizDist)) + 1
            dataIndices.append(((x_index, y_index), coorToData[hash((tube[0], tube[1]))]))

    #sort dataIndices by x coordinate, then y
    dataIndices.sort(key = (lambda x: x[0][0]*1000 + x[0][1]))

    print(dataIndices)

    print(i)

