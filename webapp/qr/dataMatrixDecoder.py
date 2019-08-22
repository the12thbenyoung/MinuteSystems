#thanks to:
#boundingRec://gist.github.com/bigsnarfdude/d811e31ee17495f82f10db12651ae82d#file-gistfile1-txt-L33k

#TODO: define MATRIX_SIZE_LOWER/UPPER_BOUND as a function of x position
#TODO: check number of contours after tube is cropped to avoid extra processing
    #on wells without tubes


import cv2
import numpy as np
from pylibdmtx.pylibdmtx import decode
from multiprocessing import Pool, Process, Queue

FILENAME = 'shpongus.jpg'

SHOW_IMAGES = 0
SHOW_RACK = 0
PRINT_CONTOUR_SIZES = 0
PRINT_MATRIX_SIZES = 0
DRAW_CONTOURS = 0

#max number of pixels of different between y coordinates for tubes to be considered in same row
SAME_ROW_THRESHOLD = 70

#number of rows and columns in rack
NUM_ROWS = 8
NUM_COLS = 12

#for cropping tube area (tube+rims/shadows) from rack
TUBE_AREA_THRESH_CONSTANT = 25
#for cropping just circular tube from tube area
TUBE_THRESH_FACTOR = 0.80
#for cropping data matrix from harris corner heatmap image
HARRIS_THRESH_FACTOR = 0.01
#for removing surrounding numbers but keeping data matrix
NUMBERS_THRESH_FACTOR = 0.30
#for thresholding matrix after it's been cropped from tube
MATRIX_THRESH_FACTOR = 0.35

#size of box used in harris corner algorithm
HARRIS_BLOCK_SIZE = 6

#typical edge lengths of contour bounding boxes of tube areas
TUBE_HEIGHT_LOWER_BOUND = 150
TUBE_HEIGHT_UPPER_BOUND = 275
TUBE_WIDTH_LOWER_BOUND = 110
TUBE_WIDTH_UPPER_BOUND = 300

#approx pixel edge length of perfectly cropped matrix
MATRIX_SIZE_LOWER_BOUND = 95
MATRIX_SIZE_UPPER_BOUND = 120

#amount to adjust NUMBERS_THRESH_FACTOR by when trying to crop matrix
NUMBERS_THRESH_INCREMENT = 0.025

#pixel dimension of kernel square
MORPHOLOGY_KERNEL_SIZE=3
#kernel for morphology
morphology_kernel = np.ones((MORPHOLOGY_KERNEL_SIZE, MORPHOLOGY_KERNEL_SIZE), np.uint8)

DILATION_KERNEL_SIZE=2
dilation_kernel = np.ones((DILATION_KERNEL_SIZE, DILATION_KERNEL_SIZE), np.uint8)

RACK_EROSION_KERNEL_SIZE=5
rack_erosion_kernel = np.ones((RACK_EROSION_KERNEL_SIZE, RACK_EROSION_KERNEL_SIZE), np.uint8)

TUBE_EROSION_KERNEL_SIZE=3
tube_erosion_kernel = np.ones((TUBE_EROSION_KERNEL_SIZE, TUBE_EROSION_KERNEL_SIZE), np.uint8)

#max number of times to run dilation
MAX_DILATE_ITERS = 10

#extra pixels added to edge of matrix crop
MATRIX_EDGE_PIXELS = 5

#left col bound of area where too-narrow contours are extended
CONTOUR_EXTEND_LOWER_X = 2800
#if contour is narrower than this number pixels, extend it to the right until it's this wide
TUBE_MIN_WIDTH = 180

#min number of contours a tube must have for it to be processed
MIN_CONTOURS_IN_MATRIX = 10

#approximate x and y coordinates of each row and column
ROW_COORDINATES = [65, 353, 640, 930, 1210, 1490, 1790, 2085]
COLUMN_COORDINATES = [90, 345, 610, 870, 1130, 1390, 1650, 1910, 2160, 2440, 2720, 2980]

badcount = 0

def show_image_small(*args):
    for (name, img) in args:
        cv2.imshow(name, cv2.resize(img, (int(img.shape[1]/5), int(img.shape[0]/5)), interpolation=cv2.INTER_AREA))
    cv2.waitKey(0)

def draw_contour_boxes(img, bounding_rect_dims):
    for dim in bounding_rect_dims:
        cv2.rectangle(img, (dim['x'], dim['y']), (dim['x'] + dim['w'], dim['y'] + dim['h']), 1, thickness=5)
    show_image_small(['rack', img])
    exit(0)

def output_data(filename, data_locations):
    with open(filename, 'w') as file:
        letters = 'ABCDEFGH'
        # file.write('row,col,value\n')
        for row in data_locations:
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

    return img_crop

# like crop_smallest_rect, but bounding rect isn't rotated and thus can be cropped directly
#param edge_pix is the number of extra pixels to crop on each sied inside bounding rect
def crop_bounding_rect(img, contour, edge_pix):
    rect = cv2.boundingRect(contour)
    x, y, w, h = rect

    img_crop = img[y+edge_pix : y+h-edge_pix, x+edge_pix : x+w-edge_pix]
    return img_crop


def crop_area_to_tube(dims, rack):
    x,y,w,h = dims['x'], dims['y'], dims['w'], dims['h']
    #crop to bounding rect around contour
    tubeArea = rack[y:y+h, x:x+w]

    #crop again to just tube, trying to cut out surrounding shadow
    tube_inv = 255 - tubeArea
    tubeContour = find_largest_contour(tube_inv, TUBE_THRESH_FACTOR)
    if tubeContour.any():
        tube = crop_bounding_rect(tubeArea, tubeContour, 5)
    else:
        return np.array([])
    
    return tube

def crop_to_harris(img, blockSize, numbers_thresh_factor, harris = np.array([])):
    #we only pass in harris if matrix is too small, in which case we dilate it
    if harris.size != 0:
        harris = cv2.dilate(harris, dilation_kernel, iterations=1)
    #otherwise matrix is too big, so find it again with larger numbers_thresh_factor
    else:
        #hopefully remove numbers while keeping matrix
        _, thr = cv2.threshold(img, numbers_thresh_factor* img.max(), 255, cv2.THRESH_BINARY)
        # cv2.imshow('thr', thr)

        #detect corners
        opening = cv2.morphologyEx(thr, cv2.MORPH_OPEN, morphology_kernel)
        harris = cv2.cornerHarris(opening, HARRIS_BLOCK_SIZE, 1, 0.00)
  
    contour = find_largest_contour(harris, HARRIS_THRESH_FACTOR)
    if contour.any():
        contourArea = cv2.contourArea(cv2.convexHull(contour)),

        #crop out matrix from tube
        matrix = crop_smallest_rect(img, contour, MATRIX_EDGE_PIXELS)

        # cv2.imshow('matrix', matrix)
        # cv2.waitKey(0)

        height, width = matrix.shape[0], matrix.shape[1]
        return matrix, harris, height, width
    else:
        return np.array([]), harris, 0, 0

def process_matrix(img, blockSize, threshFactor):
    if img.size != 0:
        numbers_thresh_factor = NUMBERS_THRESH_FACTOR
        matrix, harris, height, width = crop_to_harris(img, blockSize, NUMBERS_THRESH_FACTOR)
        if SHOW_IMAGES:
            cv2.imshow('crop', matrix)
            cv2.waitKey(0)

        if PRINT_MATRIX_SIZES:
            print(height, width)

        #if matrix is too big, raise NUMBERS_THRESH_FACTOR to try to crop out numbers 
        #and just get matrix
        iters = 0
        #if one side is too small we just want to skip to the too-small iteration below
        while all([dim > bound \
                    for dim in [height, width] \
                    for bound in [MATRIX_SIZE_UPPER_BOUND, MATRIX_SIZE_LOWER_BOUND]]) \
              and iters < MAX_DILATE_ITERS:
            numbers_thresh_factor += NUMBERS_THRESH_INCREMENT
            # print('too big', height, width)
            matrix, harris, height, width = crop_to_harris(img, \
                                                         blockSize, \
                                                         numbers_thresh_factor)
            iters += 1

        #if matrix is too small, there were likely gaps in the harris threshold that split
        #the matrix into multiple contours, so dilate harris to try to connect the split matrix
        #into one contour
        iters = 0
        while any([dim < MATRIX_SIZE_LOWER_BOUND
                   for dim in [height, width]]) and iters < MAX_DILATE_ITERS:
            matrix, harris, height, width = crop_to_harris(img, \
                                                         blockSize, \
                                                         NUMBERS_THRESH_FACTOR,
                                                         harris = harris)
            iters += 1

        #empty slots have weird aspect ratios (tall or wide) after they're processed. We 
        #can use this to filter out some of them
        if height/width < 1.5 and width/height < 1.5:

            #only otsu threshold matrix if it's cropped nicely - otherwise lighter surrounding
            #colors will mess it up
            if all([dim < MATRIX_SIZE_UPPER_BOUND and dim > MATRIX_SIZE_LOWER_BOUND
                    for dim in [height, width]]):
                matrix_blur = cv2.GaussianBlur(matrix, (5,5), 0)
                _, matrix_thr = cv2.threshold(matrix_blur, 0, 255, \
                                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                decoded_matrix = decode(matrix_thr)

            else:
                decoded_matrix = decode(matrix)

            return decoded_matrix
        else:
            print('zoinks scoob thats a bad aspect ratio')
            return None
    else:
        return None


def process_tube(tube_dict):
    global badcount
    tube_img = tube_dict['image']
    if SHOW_IMAGES:
        cv2.imshow('tube', tube_img)

    #if there aren't many contours this is probably an empty well, so don't waste time
    #trying to process it
    tube_thr = cv2.adaptiveThreshold(tube_img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                                       cv2.THRESH_BINARY,301,0)
    tube_erode = cv2.erode(tube_thr, tube_erosion_kernel, iterations=1)

    contours, _ = cv2.findContours(tube_erode, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) < MIN_CONTOURS_IN_MATRIX:
        del tube_dict['image']
        tube_dict['data'] = None
        return tube_dict

    data = process_matrix(tube_img, HARRIS_BLOCK_SIZE, HARRIS_THRESH_FACTOR)

    if not data:
        #if this failed, just run on uncropped tube
        data = decode(tube_img)
        if not data:
            #try again, but threshold tube
            x, tube_thr = cv2.threshold(tube_img, MATRIX_THRESH_FACTOR * tube_img.max(), 255, cv2.THRESH_BINARY)
            data = decode(tube_thr)
            #finally, if it really couldn't find it
            if not data:
                badcount += 1
                print('zoinks')
                cv2.imwrite(f'images/badkek{badcount}.jpg', tube_img)
    #don't need image anymore, but do want result of decode 
    del tube_dict['image']
    tube_dict['data'] = int(data[0].data) if data else None
    return tube_dict

#take dimensions (x,y,w,h) of rectangle around tube and its shadow and crop to an
#image containing a tighter rectangle around just the tube
def get_data_indices(data_locations, img=None):
    data_indices = {}

    #if we get an image, draw coordinates next to tubes (for debugging)
    if img != None:
        for data in data_locations:
            cv2.putText(img, f'{data["x"]},{data["y"]}', (data['x'], data['y']), cv2.FONT_HERSHEY_SIMPLEX, 1, 1, thickness=5)
        show_image_small(['rack', img])

    #find indices of row and column closest to each tube
    for data_loc in data_locations:
        row = np.argmin([abs(data_loc['y'] - row_coor) for row_coor in ROW_COORDINATES])
        col = np.argmin([abs(data_loc['x'] - col_coor) for col_coor in COLUMN_COORDINATES])
        #the image is sideways - here we hash (row,col) but elsewhere the hash is (col,row)
        #if things are backwards, you may have to do (row,12-col) or something like that
        data_indices[hash((row,col))] = data_loc['data']

    return data_indices

def process_rack(rack_num, filename, data_queue = None):
    badcount = 0
    tubes_found, matrices_decoded = 0,0

    img = cv2.imread(filename)
    #convert to grayscale
    rack_original = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #camera has to be at an angle to be in focus, meaning tubes on the right side are
    #smaller than their bretheren on the left. Warp perspective to correct this
    rows,cols = rack_original.shape

    source_pts = np.float32([[0,0],[0,rows-1],[cols-1,200],[cols-1,rows-200]])
    destination_pts = np.float32([[0,0],[0,rows-1],[cols-1,0],[cols-1,rows-1]])
    distort_matrix = cv2.getPerspectiveTransform(source_pts,destination_pts)
    rack_warp = cv2.warpPerspective(rack_original,distort_matrix,(cols,rows))

    rack_blur = cv2.GaussianBlur(rack_warp,(5,5),0)
    rack_thr = cv2.adaptiveThreshold(rack_blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                                       cv2.THRESH_BINARY_INV,301,TUBE_AREA_THRESH_CONSTANT)

    # erode image to try to get rid of white bridges between tubes and gaps on side of rack
    rack_erode = cv2.erode(rack_thr, rack_erosion_kernel, iterations=1)

    #open to remove noise and give smaller number of contours
    rack_open = cv2.morphologyEx(rack_thr, cv2.MORPH_OPEN, morphology_kernel, iterations=5)

    if SHOW_RACK:
        show_image_small(['rack', rack_open])
        exit(0)

    #get contours around tubes
    contours, hierarchy = cv2.findContours(rack_open, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #each set holds a group of points with similar y-coordinate - corresponding to one row in the rack
    rowLists = []
    #dict to associate (x,y) coordinate pairs with decoded outputs.
    #get hash value with 100*x + y
    coorToData = {}
    #tuples of (indices, decoded data)
    data_locations = []

    #smallest and largest x-coordinate found
    maxX = -1*float('inf')
    minX = float('inf')

    if PRINT_CONTOUR_SIZES:
        for c in contours:
            print(cv2.boundingRect(c))
            cv2.imshow('crop', crop_bounding_rect(rack_warp, c, 0))
            cv2.waitKey(0)

    #smallest non-rotated bounding rectangle for each contour
    #and filter our those which are too small
    bounding_rect_dims = list(filter(lambda c: c['w'] > TUBE_WIDTH_LOWER_BOUND and \
                                        c['w'] < TUBE_WIDTH_UPPER_BOUND and \
                                        c['h'] > TUBE_HEIGHT_LOWER_BOUND and\
                                        c['h'] < TUBE_HEIGHT_UPPER_BOUND, \
                              map(lambda c: dict(zip(('x','y','w','h'), \
                                                     cv2.boundingRect(c))), \
                                  contours)))

    #make a list of cropped tube images
    tube_images = []
    for dims in bounding_rect_dims:
        #associates tube image with its contour coordinates
        tube_dict = {
            'x': dims['x'],
            'y': dims['y']
        }
        #contours on the extreme right of the image are often too narrow and don't
        #cover the right part of the matrix. So if width is too small, extend to the right
        if dims['x'] > CONTOUR_EXTEND_LOWER_X and dims['w'] < TUBE_MIN_WIDTH:
            dims['w'] += TUBE_MIN_WIDTH - dims['w']
        tube_img = crop_area_to_tube(dims, rack_warp)
        if tube_img.size != 0:
            tube_dict['image'] = tube_img
            tube_images.append(tube_dict)
    tubes_found = len(tube_images)

    if DRAW_CONTOURS:
        draw_contour_boxes(rack_warp, bounding_rect_dims)

    #data_queue being passed signifies that this should be done w/ multiprocessing
    if data_queue:
        #create a pool of processes to decode images in parallel
        p = Pool(4)
        codes = p.map(process_tube, tube_images)
    else:
        codes = []
        for tube in tube_images:
            codes.append(process_tube(tube))

    for data_dict in codes:
        if data_dict['data']:
            matrices_decoded += 1
            data_locations.append(data_dict)

    #dict that associates hashed (x,y) position with data
    data_indices = get_data_indices(data_locations)

    #if we're passed a queue, this is being called by a (multi)process, so add to queue
    #with key filename so parent process can tell which image generated the data
    if data_queue:
        data_queue.put((rack_num, filename, data_indices, tubes_found, matrices_decoded))
        return None, None, None
    else:
        return data_indices, tubes_found, matrices_decoded

if __name__ == '__main__':
    data_locations, tubes_found, matrices_decoded = process_rack(FILENAME)
    print(data_locations)
    print(f'tubes found: {tubes_found}')
    print(f'decoded: {matrices_decoded}')
