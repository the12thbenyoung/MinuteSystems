#thanks to:
#boundingRec://gist.github.com/bigsnarfdude/d811e31ee17495f82f10db12651ae82d#file-gistfile1-txt-L33k

import cv2
import numpy as np
from pylibdmtx.pylibdmtx import decode

FILENAME = 'sidekek.jpg'

SHOW_IMAGES = 0
SHOW_RACK = 0
PRINT_CONTOUR_SIZES = 0

#max number of pixels of different between y coordinates for tubes to be considered in same row
SAME_ROW_THRESHOLD = 70

#number of rows and columns in rack
NUM_ROWS = 8
NUM_COLS = 12

#for cropping rack from original image
RACK_THRESH_FACTOR = 0.75
#for cropping tube area (tube+rims/shadows) from rack
TUBE_AREA_THRESH_FACTOR = 0.26
TUBE_AREA_THRESH_CONSTANT = 20
#for cropping just circular tube from tube area
TUBE_THRESH_FACTOR = 0.75
#for cropping data matrix from harris corner heatmap image
HARRIS_THRESH_FACTOR = 0.01
#for removing surrounding numbers but keeping data matrix
NUMBERS_THRESH_FACTOR = 0.35
#for thresholding matrix after it's been cropped from tube
MATRIX_THRESH_FACTOR = 0.35

#size of box used in harris corner algorithm
HARRIS_BLOCK_SIZE = 6

#typical edge lengths of contour bounding boxes of tube areas
EDGE_LOWER_BOUND = 170
EDGE_UPPER_BOUND = 270

#approx pixel edge length of perfectly cropped matrix
MATRIX_SIZE_LOWER_BOUND = 85
MATRIX_SIZE_UPPER_BOUND = 95

#smallest expected area of a contour containing a matrix in the harris image
MIN_MATRIX_CONTOUR_AREA = 4000

#amount to adjust HARRIS_THRESH_FACTOR by when trying to crop matrix
THRESH_INCREMENT_UP = 0.003
THRESH_INCREMENT_DOWN = 0.001

#pixel dimension of kernel square
MORPHOLOGY_KERNEL_SIZE=3
#kernel for morphology
morphologyKernel = np.ones((MORPHOLOGY_KERNEL_SIZE, MORPHOLOGY_KERNEL_SIZE), np.uint8)

DILATION_KERNEL_SIZE=2
dilationKernel = np.ones((DILATION_KERNEL_SIZE, DILATION_KERNEL_SIZE), np.uint8)

#max number of times to run dilation
MAX_DILATE_ITERS = 10

#extra pixels added to edge of matrix crop
MATRIX_EDGE_PIXELS = 4


def show_image_small(name, img):
    cv2.imshow(name, cv2.resize(img, (int(img.shape[1]/5), int(img.shape[0]/5)), interpolation=cv2.INTER_AREA))
    cv2.waitKey(0)

def outputData(filename, dataIndices):
    with open(filename, 'w') as file:
        letters = 'ABCDEFGH'
        # file.write('row,col,value\n')
        for row in dataIndices:
            file.write('{},{},{}\n'.format(letters[row[0][1]-1],row[0][0],int(row[1])))

#threshold image and find largest area contour
def find_largest_contour(img, threshFactor):
    x, thr = cv2.threshold(img, threshFactor * img.max(), 255, cv2.THRESH_BINARY)

    if SHOW_IMAGES:
        cv2.imshow('thr', thr)

    #get outer contour of data matrix
    thr = thr.astype('uint8')
    contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    areas = list(map(lambda x: cv2.contourArea(cv2.convexHull(x)), contours))
    if areas:
        max_index = np.argmax(areas)
        return contours[max_index]
    else:
        return np.array([])


#find the minAreaRect around the largest light object in img, rotate and crop
def crop_smallest_rect(img, contour, edgePix):
    rect = cv2.minAreaRect(contour)

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
    size = (x+edgePix, y+edgePix)

    # now rotated rectangle becomes vertical and we crop it
    img_crop = cv2.getRectSubPix(img_rot, size, center)

    if SHOW_IMAGES:
        cv2.imshow('crop', img_crop)
        cv2.waitKey(0)

    return img_crop

# like crop_smallest_rect, but bounding rect isn't rotated and thus can be cropped directly
#param edge_pix is the number of extra pixels to crop on each sied inside bounding rect
def crop_bounding_rect(img, contour, edge_pix):
    rect = cv2.boundingRect(contour)
    x, y, w, h = rect

    img_crop = img[y+edge_pix : y+h-edge_pix, x+edge_pix : x+w-edge_pix]
    return img_crop

def process_matrix(img, blockSize, threshFactor):
    #hopefully remove numbers while keeping matrix
    _, thr = cv2.threshold(img, NUMBERS_THRESH_FACTOR* img.max(), 255, cv2.THRESH_BINARY)

    #detect corners
    opening = cv2.morphologyEx(thr, cv2.MORPH_OPEN, morphologyKernel)
    harris = cv2.cornerHarris(opening, HARRIS_BLOCK_SIZE, 1, 0.00)
  
    contour = find_largest_contour(harris, threshFactor)
    if contour.any():
        contourArea = cv2.contourArea(cv2.convexHull(contour)),

        #if threshFactor reached 0, we haven't found the matrix, so it likely isn't there
        if threshFactor <= 0:
            return None, img

        #crop out matrix from tube
        matrix = crop_smallest_rect(img, contour, MATRIX_EDGE_PIXELS)
        height, width = matrix.shape[0], matrix.shape[1]

        ##keep raising threshold until matrix becomes sufficiently small, meaning
        ##it's been cropped to just the matrix and not the surrounding numbers
        #while any([dim > MATRIX_SIZE_UPPER_BOUND
        #          for dim in [height, width]]) and threshFactor < 1:
        #    threshFactor += THRESH_INCREMENT_UP
        #    contour = find_largest_contour(harris, threshFactor)
        #    contourArea = cv2.contourArea(cv2.convexHull(contour)),
        #    matrix = crop_smallest_rect(img, contour)
        #    height, width = matrix.shape[0], matrix.shape[1]

        #if matrix is too small, threshFactor was too high (didn't find whole matrix),
        #so lower it and try again
        iters = 0
        while any([dim < MATRIX_SIZE_LOWER_BOUND
                   for dim in [height, width]]) and iters < MAX_DILATE_ITERS:
            harris = cv2.dilate(harris, dilationKernel, iterations=1)
            contour = find_largest_contour(harris, threshFactor)
            if contour.any():
                contourArea = cv2.contourArea(cv2.convexHull(contour)),
                matrix = crop_smallest_rect(img, contour, MATRIX_EDGE_PIXELS)
                height, width = matrix.shape[0], matrix.shape[1]
                iters += 1
            else:
                break

        # cv2.imshow('matrix', matrix)
        # cv2.waitKey(0)

        #empty slots have weird aspect ratios (tall or wide) after they're processed. We 
        #can use this to filter out some of them
        if height/width < 1.5 and width/height < 1.5:
            return decode(matrix), matrix
        else:
            print('zoinks scoob thats a bad aspect ratio')
            return None, matrix
    else:
        return None, img

if __name__ == '__main__':
    tubesFound = 0
    img = cv2.imread(FILENAME)

    #convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #rackContour = find_largest_contour(gray, RACK_THRESH_FACTOR)
    ##crop to just rack 
    #rack = crop_smallest_rect(gray, rackContour, 0)
    rack = gray

    #threshold and invert to get tubes in white
    # x, rack_thr = cv2.threshold(rack, TUBE_AREA_THRESH_FACTOR * rack.max(), 255, cv2.THRESH_BINARY)
    rack_blur = cv2.GaussianBlur(rack,(5,5),0)
    rack_thr = cv2.adaptiveThreshold(rack_blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                                       cv2.THRESH_BINARY,301,TUBE_AREA_THRESH_CONSTANT)
    rack_thr = 255 - rack_thr

    if SHOW_RACK:
        show_image_small('rack_thr', rack_thr)
        exit(0)

    #get contours around tubes
    contours, hierarchy = cv2.findContours(rack_thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #each set holds a group of points with similar y-coordinate - corresponding to one row in the rack
    rowLists = []
    #dict to associate (x,y) coordinate pairs with decoded outputs.
    #get hash value with 100*x + y
    coorToData = {}
    #tuples of (indices, decoded data)
    dataIndices = []

    matricesDecoded = 0
    #smallest and largest x-coordinate found
    maxX = -1*float('inf')
    minX = float('inf')
    for contour in contours:
        #smallest non-rotated bounding rectangle
        rect = cv2.boundingRect(contour)
        x, y, w, h = rect

        if PRINT_CONTOUR_SIZES:
            print(w,h)

        #hopefully only the tube contours are this square and this big
        if all([EDGE_LOWER_BOUND < dim and EDGE_UPPER_BOUND > dim for dim in [w,h]]):
            #crop to bounding rect around contour
            tubeArea = rack[y:y+h, x:x+w]

            #crop again to just tube, trying to cut out surrounding shadow
            tube_inv = 255 - tubeArea
            tubeContour = find_largest_contour(tube_inv, TUBE_THRESH_FACTOR)
            if tubeContour.any():
                tube = crop_bounding_rect(tubeArea, tubeContour, 5)
            else:
                continue
            
            if SHOW_IMAGES:
                cv2.imshow('tube', tube)

            data, matrix = process_matrix(tube, HARRIS_BLOCK_SIZE, HARRIS_THRESH_FACTOR)

            tubesFound += 1

            if data:
                matricesDecoded += 1
            else:
                #if this failed, just run on uncropped tube
                data = decode(tube)
                if data:
                    matricesDecoded += 1
                else:
                    #try again, but threshold tube
                    x, tube_thr = cv2.threshold(tube, MATRIX_THRESH_FACTOR * tube.max(), 255, cv2.THRESH_BINARY)
                    data = decode(tube_thr)
                    if data:
                        matricesDecoded += 1
                    else:
                        print('zoinks')
                        cv2.imwrite(f'images/badkek{tubesFound}.jpg', tube)
                        # cv2.putText(img, 'no bueno', (x,y), cv2.FONT_HERSHEY_PLAIN, 5, (255,0,0), thickness=3)

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

    print(f'tubes found: {tubesFound}')
    print(f'decoded: {matricesDecoded}')
