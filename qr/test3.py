import numpy as np
import cv2
from pylibdmtx.pylibdmtx import decode
from findTubes import *

img = cv2.imread('shpongus.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

data = decode(gray)
print(data)
print(len(data))

