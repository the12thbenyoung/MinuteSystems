import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('shpongus.jpg')
rows,cols,ch = img.shape
print(rows, cols)

pts1 = np.float32([[0,0],[0,rows-1],[cols-1,200],[cols-1,rows-200]])
pts2 = np.float32([[0,0],[0,rows-1],[cols-1,0],[cols-1,rows-1]])

M = cv2.getPerspectiveTransform(pts1,pts2)
dst = cv2.warpPerspective(img,M,(cols,rows))

cv2.imwrite('warped_shpongus.jpg', dst)
plt.subplot(121),plt.imshow(img),plt.title('Input')
plt.subplot(122),plt.imshow(dst),plt.title('Output')
plt.show()
