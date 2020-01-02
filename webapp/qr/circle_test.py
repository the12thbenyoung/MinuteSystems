import cv2
import numpy as np
from dataMatrixDecoder import show_image_small

for i in range(5):
    img = cv2.imread(f'real_images/rack{i}.jpg')
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
    rack_blur_crop = rack_blur[0:750,0:750]
    circles = cv2.HoughCircles(rack_blur, cv2.HOUGH_GRADIENT, 1, 250,\
                               param1=60, param2=20, minRadius=80, maxRadius=100)
    if circles is not None:
        # convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")

        # loop over the (x, y) coordinates and radius of the circles
        for (x, y, r) in circles:
            # draw the circle in the output image, then draw a rectangle
            # corresponding to the center of the circle
            cv2.circle(rack_blur, (x, y), r, 255, 4)
            cv2.rectangle(rack_blur, (x - 5, y - 5), (x + 5, y + 5), 255, -1)
        # show_image_small(['rack', rack_blur])
        print(len(circles))
        show_image_small(['rack',rack_blur])

# for i in range(0,100,5):
#     for j in range(0,100,5):
#         circles = cv2.HoughCircles(rack_blur, cv2.HOUGH_GRADIENT, 1, 250,\
#                                    param1=i, param2=j, minRadius=75, maxRadius=90)
#         print(i,j,len(circles))
