#thanks to:
#boundingRec://gist.github.com/bigsnarfdude/d811e31ee17495f82f10db12651ae82d#file-gistfile1-txt-L33k

#TODO: define MATRIX_SIZE_LOWER/UPPER_BOUND as a function of x position
#TODO: check number of contours after tube is cropped to avoid extra processing
    #on wells without tubes


import cv2
import numpy as np
from pylibdmtx.pylibdmtx import decode
from time import sleep
from multiprocessing import Pool, Process, Queue
import os
import math

FILENAME = 'real_images/rack4.jpg'

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
TUBE_AREA_THRESH_CONSTANT = 20
#for cropping just circular tube from tube area
TUBE_THRESH_FACTOR = 0.80
#for cropping data matrix from harris corner heatmap image
HARRIS_THRESH_FACTOR = 0.01
#for removing surrounding numbers but keeping data matrix
NUMBERS_THRESH_FACTOR = 0.30
#for thresholding tube to find matrix in first failsafe
FAILSAFE_THRESH_FACTOR = 0.35
#for adaptive thresholding tube to find matrix in second failsafe
FAILSAFE_ADAPTIVE_THRESH_FACTOR = 5

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
ROW_COORDINATES = [110, 398, 686, 974, 1262, 1550, 1838, 2130] # OLD VALUES [65, 353, 640, 930, 1210, 1490, 1790, 2085]
COLUMN_COORDINATES = [47, 304, 564, 831, 1094, 1356, 1618, 1876, 2139, 2400, 2663, 2910] # OLD VALUES[90, 345, 610, 870, 1130, 1390, 1650, 1910, 2160, 2440, 2720, 2980]

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


def crop_area_to_tube(x, y, r, x_offset, rack):
    #crop to bounding rect around contour
    tubeArea = rack[y - r : y + r, x + x_offset - r : x + x_offset + r]

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
        if width!=0 and height!=0 and height/width < 1.5 and width/height < 1.5:

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
            # print('zoinks scoob thats a bad aspect ratio')
            return None
    else:
        return None

def move_along_slope(point, slope, dist, positive_x, *args):
    """return: (x,y) - point resulting from moving dist along line
       point: (x,y) coordinates of starting point
       slope: slope of target line
       dist: distance in pixels to move along line
       positive_x: True if positive x-direction, False if negative
       args: slope, dist, positive_x again to move along two lines"""
    x, y = point
    #move dist px in target direction:
    #l = dist
    #dy = m*dx
    #dy^2 + dx^2 = l^2
    #(m*dx)^2 + dx^2 = l^2
    #(m^2+1)dx^2 = l^2
    #dx = sqrt(l^2/(m^2+1))
    dx = math.sqrt(dist**2 / (slope**2 + 1))
    dy = slope*dx
    if positive_x:
        new_x = int(x+dx)
        new_y = int(y+dy)
    else:
        new_x = int(x-dx)
        new_y = int(y-dy)
    if not args:
        return new_x, new_y
    elif len(args) == 3:
        #move again along second line
        slope, dist, positive_x = args
        dx = math.sqrt(dist**2 / (slope**2 + 1))
        dy = slope*dx
        if positive_x:
            new_x = int(new_x+dx)
            new_y = int(new_y+dy)
        else:
            new_x = int(new_x-dx)
            new_y = int(new_y-dy)
        return new_x, new_y
    else:
        raise Exception("Improper number of extra arguments passed. Required: 0 or 3 (slope, dist, positive_x)")

def hough_process_tube(tube_dict):
    gray = tube_dict['image']
    cv2.imwrite(f'test.jpg', gray)
    del tube_dict['image']
    if gray.shape[0] == 0 or gray.shape[1] == 0:
        return tube_dict
    tube_blur = cv2.GaussianBlur(gray,(5,5),0)

    # cv2.imshow('tube_blur', tube_blur)
    # cv2.waitKey(0)

    # decoded_matrix = decode(gray)
    # print(decoded_matrix[0].data)

    rho_diff = 20
    theta_diff = 0.1
    #separate into groups by angle - should be two groups of lines close together
    line_groups = []
    canny_upper_threshold = 120
    while(len(line_groups) < 2 and canny_upper_threshold > 80):
        canny_upper_threshold -= 5
        edges = cv2.Canny(tube_blur, 30, canny_upper_threshold, apertureSize=3)

        # cv2.imshow('edges', edges)
        # cv2.waitKey(0)

        lines = cv2.HoughLines(edges,1,np.pi/90,50)
        if lines is None:
            continue
        #get rid of extra lists
        lines = [l[0] for l in lines]

        # for line in lines:
        #     rho,theta = line
        #     a = np.cos(theta)
        #     b = np.sin(theta)
        #     x0 = a*rho
        #     y0 = b*rho
        #     x1 = int(x0 + 1000*(-b))
        #     y1 = int(y0 + 1000*(a))
        #     x2 = int(x0 - 1000*(-b))
        #     y2 = int(y0 - 1000*(a))

        #     cv2.line(tube_blur,(x1,y1),(x2,y2),(0,0,255),2)

        # cv2.imshow('lines', tube_blur)
        # cv2.waitKey(0)

        for line in lines:
            found_group = False
            if not line_groups:
                line_groups.append([line])
            else:
                for group in line_groups:
                    if abs(line[0] - np.mean([l[0] for l in group])) < rho_diff and \
                            abs(line[1] - np.mean([l[1] for l in group])) < theta_diff:
                        group.append(line)
                        found_group = True
                        break
                if not found_group:
                    line_groups.append([line])

    if len(line_groups) >= 2:
        #we found enough lines
        #take the two largest groups with angle difference around pi/2
        avg_thetas = [np.mean([line[1] for line in group]) for group in line_groups]
        avg_rhos = [np.mean([line[0] for line in group]) for group in line_groups]
        index_pair_candidates = [(i,j) for i in range(len(line_groups))
                                       for j in range(i)
                                       if abs(abs(avg_thetas[i] - avg_thetas[j]) - math.pi/2) < 0.2]
        if not index_pair_candidates:
            return tube_dict
        bounding_index_pair = max(index_pair_candidates,
                                  key=lambda t: len(line_groups[t[0]]) + len(line_groups[t[1]]))
    else:
        return tube_dict

       #get best line in each of the two chosen groups
    bounding_lines = []
    for bounding_group_index in bounding_index_pair:
        if avg_rhos[bounding_group_index] > 0:
            if 0 <= avg_thetas[bounding_group_index] < math.pi/2:
                if 0 <= avg_rhos[bounding_group_index] < math.sqrt(tube_blur.shape[0]**2 + tube_blur.shape[1]**2)/2:
                    #line is on inner side (upper left) of picture, so to choose line furthest from
                    #matrix edge choose innermost line (smallest row)
                    bounding_line_test = lambda line: -line[0]
                else:
                    #line on outer side of picture, choose outermost line
                    bounding_line_test = lambda line: line[0]
            else: #avg_thetas[bounding_group_index] >= math.pi/2
                #always choose line with largest rho
                    bounding_line_test = lambda line: line[0]
        else: #avg_rhos[bounding_group_index] <= 0:
            #also always use largest rho
            bounding_line_test = lambda line: line[0]

        bounding_lines.append(max(line_groups[bounding_group_index],
                                  key=bounding_line_test))

    m = [None,None]
    b = [None,None]
    #find intersection of two bounding lines
    rho1, theta1 = bounding_lines[0]
    rho2, theta2 = bounding_lines[1]
    sin_theta1 = np.sin(theta1)
    sin_theta2 = np.sin(theta2)
    m[0] = -np.cos(theta1)/(sin_theta1 if sin_theta1 != 0 else 1e-10)
    m[1] = -np.cos(theta2)/(sin_theta2 if sin_theta2 != 0 else 1e-20)
    b[0] = rho1/(sin_theta1 if sin_theta1 != 0 else 1e-20)
    b[1] = rho2/(sin_theta2 if sin_theta2 != 0 else 1e-20)

    #solve system to find intersection of lines
    x_intersect = (b[0]-b[1])/(m[1]-m[0])
    y_intersect = m[0]*x_intersect + b[0]

    MATRIX_LENGTH = 110
    CORNER_OFFSET = 10
    matrix_directions = [None, None]
    #find intersections of lines with walls of image
    for i in range(2):
        #horizontal line
        if m[i] == 0:
            #move in direction of longer line segment
            matrix_direction = tube_blur.shape[1] - x_intersect > x_intersect
        else:
            #otherwise, figure out which sides of the picture the line intersects with.
            #intersection with top boundary (y=0)
            top_intersect_x = -b[i]/m[i]
            bottom_intersect_x = (tube_blur.shape[0] - b[i])/m[i]
            #move in direction of further distance from line intersection to picture edge
            try:
                left_intersect = max(d for d in [top_intersect_x, bottom_intersect_x, 0] \
                                     if d <= x_intersect)
                right_intersect = min(d for d in [top_intersect_x, bottom_intersect_x, tube_blur.shape[1]] \
                                      if d >= x_intersect)
            except ValueError:
                print(x_intersect, top_intersect_x, bottom_intersect_x, tube_blur.shape[1])
                return tube_dict
            matrix_directions[i] = right_intersect - x_intersect > x_intersect - left_intersect

    #step a bit away from initial corner to ensure whole matrix is captured
    intersect_corner = move_along_slope((x_intersect, y_intersect), \
                                         m[0], CORNER_OFFSET, not matrix_directions[0],
                                         m[1], CORNER_OFFSET, not matrix_directions[1])
    #move to adjacent corners, overshoot a bit
    adj_corner0 = move_along_slope(intersect_corner,
                                   m[0], MATRIX_LENGTH, matrix_directions[0])
    adj_corner1 = move_along_slope(intersect_corner,
                                   m[1], MATRIX_LENGTH, matrix_directions[1])
    #move to corner opposite initial corner
    far_corner = move_along_slope(adj_corner0,
                                  m[1], MATRIX_LENGTH, matrix_directions[1])

    #warp rectangle to crop from image
    src_points = np.array([intersect_corner, adj_corner0, adj_corner1, far_corner])
    matrix_rect = cv2.minAreaRect(src_points)
    box = cv2.boxPoints(matrix_rect).astype("float32")
    width = int(matrix_rect[1][0])
    height = int(matrix_rect[1][1])
    distorted_points = np.array([[0, height-1],
                                 [0, 0],
                                 [width-1, 0],
                                 [width-1, height-1]]).astype("float32")
    M = cv2.getPerspectiveTransform(box, distorted_points)
    warped = cv2.warpPerspective(gray, M, (width, height))

    _, warped_thresh = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    warped_thresh_open = rack_open = cv2.morphologyEx(warped_thresh, cv2.MORPH_CLOSE, morphology_kernel, iterations=3)
    _, pure_thresh = cv2.threshold(warped, 0.6*warped.max(), 255, cv2.THRESH_BINARY)

    test = pure_thresh - warped_thresh_open
    # cv2.imshow('thresh', pure_thresh)
    # cv2.imshow('open', warped_thresh)
    # cv2.imshow('test', test)
    # cv2.waitKey(0)

    decoded_matrix = decode(pure_thresh)
    #don't need image anymore, but do want result of decode
    try:
        tube_dict['data'] = int(decoded_matrix[0].data) if decoded_matrix else None
    except ValueError:
        cv2.imshow('bad', tube_dict['image'])
        cv2.waitKey(0)
    return tube_dict


def harris_process_tube(tube_dict):
    global badcount
    tube_img = tube_dict['image']
    if SHOW_IMAGES:
        cv2.imshow('tube', tube_img)

    if not tube_img.shape or tube_img.shape[0] < 50 or tube_img.shape[1] < 50:
        del tube_dict['image']
        tube_dict['data'] = None
        return tube_dict

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
            #try again, but adaptive threshold
            rack_blur = cv2.GaussianBlur(tube_img,(5,5),0)
            tube_thr = cv2.adaptiveThreshold(rack_blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                                             cv2.THRESH_BINARY_INV,301,\
                                             FAILSAFE_ADAPTIVE_THRESH_FACTOR)
            data = decode(tube_thr)
            if not data:
                #try again, but threshold tube
                _, tube_thr = cv2.threshold(tube_img, FAILSAFE_THRESH_FACTOR * tube_img.max(), 255, cv2.THRESH_BINARY)
                data = decode(tube_thr)
                #finally, if it really couldn't find it
                if not data:
                    badcount += 1
                    cv2.imwrite(f'images/badkek{badcount}.jpg', tube_img)
    #don't need image anymore, but do want result of decode
    try:
        tube_dict['data'] = int(data[0].data) if data else None
    except ValueError:
        cv2.imshow('bad', tube_dict['image'])
        cv2.waitKey(0)
        exit(0)
    del tube_dict['image']
    return tube_dict

#take dimensions (x,y,w,h) of rectangle around tube and its shadow and crop to an
#image containing a tighter rectangle around just the tube
def get_data_indices(data_locations, img=None):
    matrices_decoded = 0
    data_indices = {}

    #if we get an image, draw coordinates next to tubes (for debugging)
    if img is not None:
        for data in data_locations:
            cv2.putText(img, f'{data["x"]},{data["y"]}', (data['x'], data['y']), cv2.FONT_HERSHEY_SIMPLEX, 1.5, 1, thickness=5)

    #find indices of row and column closest to each tube
    for data_loc in data_locations:
        row = np.argmin([abs(data_loc['y'] - row_coor) for row_coor in ROW_COORDINATES])
        col = np.argmin([abs(data_loc['x'] - col_coor) for col_coor in COLUMN_COORDINATES])
        #the image is sideways - here we hash (row,col) but elsewhere the hash is (col,row)
        #if things are backwards, you may have to do (row,12-col) or something like that
        if not data_indices.get(hash((7-row,col))):
            data_indices[hash((7-row,col))] = data_loc['data']
            matrices_decoded += 1

    return data_indices, matrices_decoded

def process_rack(rack_num, filename, data_queue=None):
    badcount = 0
    tubes_found = 0

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

    if SHOW_RACK:
        x = 2920
        y = 2116-2*290+30
        for i in range(4):
            cv2.rectangle(rack_warp, (x-2*i, y - 290*i), (x-2*i+200,y-290*i+200), 1, thickness=10)
        show_image_small(['rack', rack_warp])
        exit(0)

    tube_images = []
    circles = cv2.HoughCircles(rack_blur, cv2.HOUGH_GRADIENT, 1, 250,\
                               param1=60, param2=20, minRadius=80, maxRadius=100)
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x,y,r) in circles:
            r = r + 5
            #crop out rectangle surrounding tube
            #tubes on left side tend to have circle centered left of matrix and vice versa
            x_offset = (x - 1575)//1000
            # cv2.imshow('rack', rack_warp[y - r : y + r, x + x_offset - r : x + x_offset + r])
            tube_img = crop_area_to_tube(x, y, r, x_offset, rack_warp)
            # cv2.imshow('tube_img', tube_img)
            # cv2.waitKey(0)
            if tube_img.size != 0:
                tube_images.append(
                    {
                        'x': x,
                        'y': y,
                        'image': tube_img
                    }
                )
            # cv2.rectangle(rack_warp, (x+x_offset-r, y-r), (x+x_offset+r,y+r), 1, thickness=5)
            # cv2.imshow('rack', rack_warp[y - r : y + r, x + x_offset - r : x + x_offset + r])
            # cv2.waitKey(0)

        # show_image_small(['rack',rack_warp])
        # exit(0)

    #add middle four tubes in right col manually, since these sometimes don't contrast
    #with opening in rack
    bottom_x = 2919
    bottom_y = 1566
    for i in range(8):
        this_x = bottom_x - 2*i
        this_y = bottom_y - 290*i
        tube_dict = {
            'x': this_x,
            'y': this_y,
            'image': rack_warp[this_y : this_y + 200, this_x : this_x + 200]
        }
        tube_images.append(tube_dict)

    tubes_found = len(tube_images)

    #data_queue being passed signifies that this should be done w/ multiprocessing
    if False:
        #create a pool of processes to decode images in parallel
        p = Pool(2)
        codes = p.map(hough_process_tube, tube_images)
        p.close()
        p.join()
    else:
        codes = []
        for tube in tube_images:
            data = hough_process_tube(tube)
            print(data)
            codes.append(data)

    data_locations = [data_dict for data_dict in codes if data_dict.get('data')]

    #dict that associates hashed (x,y) position with data
    data_indices, matrices_decoded = get_data_indices(data_locations)

    if data_queue:
        data_queue.put((rack_num, filename, data_indices, tubes_found, matrices_decoded))
        return None, None, None
    else:
        return data_indices, tubes_found, matrices_decoded

if __name__ == '__main__':
    data_locations, tubes_found, matrices_decoded = process_rack(0, FILENAME)
    print(data_locations)
    print(f'tubes found: {tubes_found}')
    print(f'decoded: {matrices_decoded}')
