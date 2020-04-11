import numpy as np
import cv2
import matplotlib.pyplot as plt

def implot(img):
  plt.figure(figsize=(12,10))
  plt.imshow(img, cmap='gray')
  plt.show()


raw_horse = cv2.imread('data/horse.jpg')
raw_rainbow = cv2.imread('data/rainbow.jpg')
raw_bricks = cv2.imread('data/bricks.jpg')

horse = cv2.cvtColor(raw_horse, cv2.COLOR_BGR2RGB)
rainbow = cv2.cvtColor(raw_rainbow, cv2.COLOR_BGR2RGB)
bricks = cv2.cvtColor(raw_bricks, cv2.COLOR_BGR2RGB)


color = ('b', 'g', 'r')
# for i, col in enumerate(color):
#   histr = cv2.calcHist([raw_bricks], [i], None, [256], [0,256])
#   plt.plot(histr, color = col)
#   plt.xlim([0, 256])
# plt.title('Hist Baby')
# plt.show()


mask = np.zeros((raw_rainbow.shape[:2]), np.uint8)
mask[300:400, 100:400] = 255

raw_masked_rainbow = cv2.bitwise_and(raw_rainbow, raw_rainbow, mask=mask)
masked_rainbow = cv2.bitwise_and(rainbow, rainbow, mask=mask)

# implot(masked_rainbow)

hist_mask_values_red = cv2.calcHist([raw_rainbow], channels=[2], mask=mask, histSize=[256], ranges=[0, 256])
# plt.plot(hist_mask_values_red)
# plt.show()

gorilla = cv2.imread('data/gorilla.jpg', 0)
implot(gorilla)
histr = cv2.calcHist([gorilla], [0], None, [256], [0,256])
plt.plot(histr)
plt.show()

eq_gorilla = cv2.equalizeHist(gorilla)
eq_hist = cv2.calcHist([eq_gorilla], channels=[0], mask=None, histSize=[256], ranges=[0, 256])
plt.plot(eq_hist)
plt.show()