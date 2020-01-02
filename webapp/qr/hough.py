import cv2
import numpy as np
import math

canny_upper_threshold = 90
for pic in range(1):
    img = cv2.imread('images/badkek4.jpg')
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # _, thr = cv2.threshold(gray, 0.4 * gray.max(), 255, cv2.THRESH_BINARY)
    rack_blur = cv2.GaussianBlur(gray,(5,5),0)
    # thr = cv2.adaptiveThreshold(rack_blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
    #                                        cv2.THRESH_BINARY,71,0)

    # cv2.imshow('thr', thr)
    # cv2.waitKey(0)

    # # KERNEL_SIZE = 3
    # kernel = np.ones((3, 3), np.uint8)
    # opening = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel)

    #0-125: smallest
    #125- : largest
    edges = cv2.Canny(rack_blur, 30, canny_upper_threshold, apertureSize=3)

    cv2.imshow('edges', edges)
    cv2.waitKey(0)

    lines = cv2.HoughLines(edges,1,np.pi/90,50)
    #get rid of extra lists
    lines = [l[0] for l in lines]

    rho_diff = 20
    theta_diff = 0.1
    #separate into groups by angle - should be two groups of lines close together
    line_groups = []
    while(len(line_groups) < 2 and canny_upper_threshold > 80):
        canny_upper_threshold -= 5
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

    print(m)
    print(b)
    #solve system to find intersection of lines
    x_intersect = (b[0]-b[1])/(m[1]-m[0])
    y_intersect = m[0]*x_intersect + b[0]

    #for each line, hold True to move in positive x-direction
    matrix_directions = [None,None]
    #find intersections of lines with walls of image
    for i in range(2):
    #horizontal line
    if m[i] == 0:
        #move in direction of longer line segment
        matrix_directions[i] = rack_blur.shape[1] - x_intersect > x_intersect
    else:
        #otherwise, figure out which sides of the picture the line intersects with.
        #intersection with top boundary (y=0)
        top_intersect_x = -b[i]/m[i]
        bottom_intersect_x = (rack_blur.shape[0] - b[i])/m[i]
        #move in direction of further distance from line intersection to picture edge
        left_intersect = min(top_intersect_x, bottom_intersect_x, 0)
        right_intersect = max(top_intersect_x, bottom_intersect_x, rack_blur.shape[1])
        matrix_directions[i] = right_intersect - x_intersect > x_intersect - left_intersect


    numLines = 0
    for line in bounding_lines:
        print(line)
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

    print(x_intersect, y_intersect)
    cv2.circle(img, (x_intersect, y_intersect), 10, (0,255,0), thickness=-1)
    cv2.imshow('lines', img)
    cv2.waitKey(0)
