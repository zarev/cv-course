import numpy as np
import cv2
import matplotlib.pyplot as plt

def implot(img):
  plt.figure(figsize=(12,10))
  plt.imshow(img, cmap='gray')
  plt.show()


img = cv2.imread('data/sudoku.jpg')

sobelxy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize = 5)

implot(sobelxy)

kernel = np.ones((2,2), dtype=np.uint8)

lap = cv2.Laplacian(img, cv2.CV_64F)

implot(lap)

blended = cv2.addWeighted(sobelxy, 0.5, lap, 0.5, 0)

implot(blended)

