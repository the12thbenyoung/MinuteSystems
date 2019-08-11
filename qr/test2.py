import numpy as np
import cv2

from pylibdmtx.pylibdmtx import decode
from os import listdir
from multiprocessing import Pool, Process, Queue
from findTubes import *

NUMBERS_THRESH_FACTOR = 0.35

def cropToHarris(img, blockSize, numbersThreshFactor, harris = np.array([])):
    #we only pass in harris if matrix is too small, in which case we dilate it
    if harris.size != 0:
        harris = cv2.dilate(harris, dilationKernel, iterations=1)
    #otherwise matrix is too big, so find it again with larger numbersThreshFactor
    else:
        #hopefully remove numbers while keeping matrix
        _, thr = cv2.threshold(img, numbersThreshFactor* img.max(), 255, cv2.THRESH_BINARY)
        cv2.imshow('thr', thr)

        #detect corners
        opening = cv2.morphologyEx(thr, cv2.MORPH_OPEN, morphologyKernel)
        harris = cv2.cornerHarris(opening, HARRIS_BLOCK_SIZE, 1, 0.00)
  
    contour = find_largest_contour(harris, HARRIS_THRESH_FACTOR)
    if contour.any():
        contourArea = cv2.contourArea(cv2.convexHull(contour)),

        #crop out matrix from tube
        matrix = crop_smallest_rect(img, contour, MATRIX_EDGE_PIXELS)

        cv2.imshow('matrix', matrix)
        cv2.waitKey(0)

        height, width = matrix.shape[0], matrix.shape[1]
        return matrix, harris, height, width
    else:
        return np.array([]), harris, 0, 0


def process_matrix(img, blockSize, threshFactor):
    if img.size != 0:
        numbersThreshFactor = NUMBERS_THRESH_FACTOR
        matrix, harris, height,  width = cropToHarris(img, blockSize, NUMBERS_THRESH_FACTOR)

        #if matrix is too big, raise NUMBERS_THRESH_FACTOR to try to crop out numbers 
        #and just get matrix
        iters = 0
        #if one side is too small we just want to skip to the too-small iteration below
        while all([dim > bound \
                    for dim in [height, width] \
                    for bound in [MATRIX_SIZE_UPPER_BOUND, MATRIX_SIZE_LOWER_BOUND]]) \
              and iters < MAX_DILATE_ITERS:
            numbersThreshFactor += NUMBERS_THRESH_INCREMENT
            print('too big', height, width)
            matrix, harris, height, width = cropToHarris(img, \
                                                         blockSize, \
                                                         numbersThreshFactor)
            iters += 1

        #if matrix is too small, there were likely gaps in the harris threshold that split
        #the matrix into multiple contours, so dilate harris to try to connect the split matrix
        #into one contour
        iters = 0
        while any([dim < MATRIX_SIZE_LOWER_BOUND
                   for dim in [height, width]]) and iters < MAX_DILATE_ITERS:
            matrix, harris, height, width = cropToHarris(img, \
                                                         blockSize, \
                                                         NUMBERS_THRESH_FACTOR,
                                                         harris = harris)
            iters += 1

        #empty slots have weird aspect ratios (tall or wide) after they're processed. We 
        #can use this to filter out some of them
        if height/width < 1.5 and width/height < 1.5:
            if PRINT_CONTOUR_SIZES:
                print(height, width)

            #only otsu threshold matrix if it's cropped nicely - otherwise lighter surrounding
            #colors will mess it up
            if all([dim < MATRIX_SIZE_UPPER_BOUND and dim > MATRIX_SIZE_LOWER_BOUND
                    for dim in [height, width]]):
                matrix_blur = cv2.GaussianBlur(matrix, (5,5), 0)
                _, matrix_thr = cv2.threshold(matrix_blur, 0, 255, \
                                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                decoded_matrix = decode(matrix_thr)

                #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ALERT TAKE THIS OUT ALERT$$$$$$$$$$$$$$$$$$$$$$$
                decoded_matrix1 = decode(matrix)
                #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ALERT TAKE THIS OUT ALERT$$$$$$$$$$$$$$$$$$$$$$$

                if decoded_matrix1 and not decoded_matrix:
                    print('!!!!!!!!!!!!!!!!!!BAD!!!!!!!!!!!!!!!!!!!!')
                if decoded_matrix and not decoded_matrix1:
                    print('!!!!!!!!!!!!!!!!!!GOOD!!!!!!!!!!!!!!!!!!!!')
            else:
                decoded_matrix = decode(matrix)

            # cv2.imshow('matrix', matrix)
            # cv2.imshow('matrix_thr', matrix_thr)
            # cv2.waitKey(0)

            return decoded_matrix, matrix_thr
        else:
            print('zoinks scoob thats a bad aspect ratio')
            return None, matrix
    else:
        return None, img

if __name__ == '__main__':
    img = cv2.imread('images/badkek4.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    data, matrix = process_matrix(gray, HARRIS_BLOCK_SIZE, HARRIS_THRESH_FACTOR)
    print(data)
