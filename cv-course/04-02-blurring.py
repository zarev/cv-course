import numpy as np
import cv2
import matplotlib.pyplot as plt

waman = cv2.imread('cover.jpg', 0)
# waman = cv2.cvtColor(waman, cv2.COLOR_BGR2GRAY)

def load_img():
    img = cv2.imread('cover.jpg', 0).astype(np.float32) // 255
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

plt.figure(1)
plt.imshow(waman, cmap='gray')
plt.show()

gamma = 1/4

gam = np.power(waman, gamma)

plt.figure(1)
plt.imshow(gam, cmap='gray')
plt.show()

kernel = np.ones(shape=(5,5), dtype=np.float32)/25

dst = cv2.filter2D(gam, -1, kernel)

plt.figure(1)
plt.imshow(dst, cmap='gray')
plt.show()

gauss = cv2.GaussianBlur(gam, (5,5), 10)

plt.figure(1)
plt.imshow(dst, cmap='gray')
plt.show()

med = cv2.medianBlur(waman, 5)

plt.figure(1)
plt.imshow(med, cmap='gray')
plt.show()