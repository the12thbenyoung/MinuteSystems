"""
Simply display the contents of the webcam with optional mirroring using OpenCV 
via the new Pythonic cv2 interface.  Press <esc> to quit.
"""

import cv2
import numpy as np


# def show_webcam(mirror=False):
#     cam = cv2.VideoCapture(1)
#     while True:
#         ret_val, img = cam.read()
#         if mirror: 
#             img = cv2.flip(img, 1)
#         cv2.imshow('my webcam', img)
#         if cv2.waitKey(1) == 27: 
#             break  # esc to quit
#     cv2.destroyAllWindows()

cap = cv2.VideoCapture(1)
while(1):
    ret, frame = cap.read()
    #print(height)
    #cv2.imshow("Cropped Image", crop_img)
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


