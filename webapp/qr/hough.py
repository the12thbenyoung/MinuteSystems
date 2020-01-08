import cv2
import numpy as np
import math
from pylibdmtx.pylibdmtx import decode

MORPHOLOGY_KERNEL_SIZE=3
morphology_kernel = np.ones((MORPHOLOGY_KERNEL_SIZE, MORPHOLOGY_KERNEL_SIZE), np.uint8)
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


def process_tube(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    rack_blur = cv2.GaussianBlur(gray,(5,5),0)

    # cv2.imshow('rack_blur', rack_blur)
    # cv2.imshow('gray', gray)
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
        edges = cv2.Canny(rack_blur, 30, canny_upper_threshold, apertureSize=3)

        # cv2.imshow('edges', edges)
        # cv2.waitKey(0)

        lines = cv2.HoughLines(edges,1,np.pi/90,50)
        if lines is None:
            continue
        #get rid of extra lists
        lines = [l[0] for l in lines]
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
        bounding_index_pair = max([(i,j) for i in range(len(line_groups))
                                         for j in range(i)
                                   if abs(abs(avg_thetas[i] - avg_thetas[j]) - math.pi/2) < 0.2],
                                  key=lambda t: len(line_groups[t[0]]) + len(line_groups[t[1]]))
    else:
        return None

       #get best line in each of the two chosen groups
    bounding_lines = []
    for bounding_group_index in bounding_index_pair:
        if avg_rhos[bounding_group_index] > 0:
            if 0 <= avg_thetas[bounding_group_index] < math.pi/2:
                if 0 <= avg_rhos[bounding_group_index] < math.sqrt(rack_blur.shape[0]**2 + rack_blur.shape[1]**2)/2:
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
            matrix_direction = rack_blur.shape[1] - x_intersect > x_intersect
        else:
            #otherwise, figure out which sides of the picture the line intersects with.
            #intersection with top boundary (y=0)
            top_intersect_x = -b[i]/m[i]
            bottom_intersect_x = (rack_blur.shape[0] - b[i])/m[i]
            #move in direction of further distance from line intersection to picture edge
            left_intersect = max(d for d in [top_intersect_x, bottom_intersect_x, 0] \
                                 if d <= x_intersect)
            right_intersect = min(d for d in [top_intersect_x, bottom_intersect_x, rack_blur.shape[1]] \
                                  if d >= x_intersect)
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

    # test = pure_thresh - warped_thresh_open
    cv2.imshow('thresh', pure_thresh)
    cv2.imshow('open', warped_thresh)
    # cv2.imshow('test', test)
    cv2.waitKey(0)

    decoded_matrix = decode(pure_thresh)
    print(decoded_matrix[0].data)
    decoded_matrix = decode(warped_thresh)
    print(decoded_matrix[0].data)

    numLines = 0
    for line in bounding_lines:
        rho,theta = line
        numLines += 1
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))

        cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)

    cv2.circle(img, (x_intersect, y_intersect), 10, (0,255,0), thickness=-1)
    cv2.imshow('lines', img)
    cv2.waitKey(0)
if __name__ == '__main__':
    IMG_PATH = 'test.jpg'
    process_tube(IMG_PATH)
