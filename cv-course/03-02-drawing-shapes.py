import numpy as np
import cv2
import matplotlib.pyplot as plt

blank_img = np.zeros(shape=(512,512,3), dtype=np.int16)
img = cv2.imread('cover.jpg')

cv2.rectangle(img, pt1=(100,100), pt2=(300,300), color=(0,255,0), thickness=1)
cv2.circle(img, center=(100,100), radius=10, color = (0,255,0), thickness=-1)
cv2.circle(img, center=(300,300), radius=10, color = (0,255,0), thickness=-1)
cv2.circle(img, center=(300,100), radius=10, color = (0,255,0), thickness=-1)
cv2.circle(img, center=(100,300), radius=10, color = (0,255,0), thickness=-1)
plt.figure(1)
plt.imshow(img)
plt.show()
