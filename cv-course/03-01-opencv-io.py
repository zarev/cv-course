import numpy as numpy
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('cover.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
resize_img = cv2.resize(img, (0, 0), img, 0.5, 0.5)
flipped = cv2.flip(img, 1)
print (img.shape)

while True:
  cv2.imshow('img', img)
  if cv2.waitKey(1) & 0xFF == 27: break
    
cv2.destroyAllWindows()


# plt.figure(1)
# plt.imshow(resize_img)
# plt.show()

