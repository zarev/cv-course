import numpy as np
import cv2
import matplotlib.pyplot as plt

waman = cv2.imread('cover.jpg', 0)
# waman = cv2.cvtColor(waman, cv2.COLOR_BGR2GRAY)

plt.figure(1)
plt.imshow(waman, cmap='gray')
plt.show()

_, thresh = cv2.threshold(waman, waman.max()//2, waman.max(), cv2.THRESH_BINARY)

plt.figure(1)
plt.imshow(thresh, cmap='gray')
plt.show()

thresh2 = cv2.adaptiveThreshold(
              waman, waman.max(), cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 8)

plt.figure(1)
plt.imshow(thresh2, cmap='gray')
plt.show()

blended = cv2.addWeighted(thresh, 0.6, thresh2, 0.4, 0)

plt.figure(1)
plt.imshow(blended, cmap='gray')
plt.show()

