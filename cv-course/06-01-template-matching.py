import numpy as np
import cv2
import matplotlib.pyplot as plt
import helpers as hlp

dog = hlp.imread('data/sammy.jpg')
face = hlp.imread('data/sammy_face.jpg')

methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
           'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

for m in methods:
  full_cp = dog.copy()
  method = eval(m)

  # template matching
  res = cv2.matchTemplate(full_cp, face, method)

  _, _, min_loc, max_loc = cv2.minMaxLoc(res)

  top_left = max_loc
  if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED] : top_left = min_loc

  height, width, _ = face.shape

  bottom_right = (top_left[0] + width, top_left[1] + height)

  cv2.rectangle(full_cp, top_left, bottom_right, (255,0,0), 5)

  plt.subplot(121)
  plt.imshow(res)
  plt.title('Heatmap')

  plt.subplot(122)
  plt.imshow(full_cp)
  plt.title('Detections')
  plt.suptitle(m)

  plt.show()
# match = eval(methods[1])
# res = cv2.matchTemplate(dog, face, match)
# implot(res)