import numpy as np
import cv2
import matplotlib.pyplot as plt

def implot(img):
  plt.figure(figsize=(12,10))
  plt.imshow(img, cmap='gray')
  plt.show()


img = np.zeros((600, 600))
font = cv2.FONT_HERSHEY_TRIPLEX
cv2.putText(img, text = 'ABCDE', org=(50, 300), 
            fontFace=font, fontScale=5, color=(255,255,255))


kernel = np.ones((2,2), dtype=np.uint8)

# eroded = cv2.erode(img, kernel, iterations=5)

noise = np.random.randint(0, 2, (600,600)) * 255

noised_img = img + noise

# implot(noised_img)

opening = cv2.morphologyEx(noised_img, cv2.MORPH_OPEN, kernel)

# implot(opening)

black_noise = np.random.randint(0, 2, (600,600)) * -255

noised_img = img + black_noise

noised_img[noised_img == -255] = 0

implot(noised_img)

grad = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)

implot(grad)
