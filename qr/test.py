import matplotlib.pyplot as plt
import numpy as np
import cv2
from findTubes import process_matrix, find_largest_contour, crop_smallest_rect

from pylibdmtx.pylibdmtx import decode
from os import listdir
from multiprocessing import Pool, Process, Queue

def show_image_small(name, img):
    cv2.imshow(name, cv2.resize(img, (int(img.shape[1]/5), int(img.shape[0]/5)), interpolation=cv2.INTER_AREA))
    cv2.waitKey(0)

def decodeMatrix(img):
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return decode(img)

def runRack(dirpath, q):
    imgs = [cv2.imread(f'{dirpath}/{imgpath}') for imgpath in listdir(dirpath)]*3

    p = Pool(4)
    codes = p.map(decodeMatrix, imgs)
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>.hi there<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    d = {dirpath: codes}
    q.put(d)

img = cv2.imread('topkek.jpg')
img2 = img.copy()
paths = ['images', 'images2', 'images3']
q = Queue()
for path in paths:
    Process(target=runRack, args=(path,q)).start()

print(q.get())
print(q.get())
print(q.get())



# for img in imgs:
#     print(decodeMatrix(img))
# blur = cv2.GaussianBlur(gray,(5,5),0)

# TUBE_AREA_THRESH_CONSTANT = 0
# tube_thr = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
#                                  cv2.THRESH_BINARY,33,TUBE_AREA_THRESH_CONSTANT)
# print(decode(tube_thr))

# cv2.imshow('thr', tube_thr)
# cv2.waitKey(0)


