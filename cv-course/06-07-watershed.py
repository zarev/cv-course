import numpy as np
import cv2
import matplotlib.pyplot as plt
import helpers

img = helpers.imread('data/pennies.jpg', 0)

# median blur
blurred = cv2.medianBlur(img,27)
# gray
gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
# binary threshold
ret, sep_tresh = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY_INV)
# noise removal
kernel = np.ones((3,3), np.uint8)
opening = cv2.morphologyEx(sep_tresh, cv2.MORPH_OPEN, kernel, iterations=2)
bg = cv2.dilate(opening, kernel, iterations=3)
# distance transform
dist_trans = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
ret, fg = cv2.threshold(dist_trans, 0.7*dist_trans.max(), 255, 0)

fg = np.uint8(fg)
missing = cv2.subtract(bg, fg)

ret, markers = cv2.connectedComponents(fg)

markers = markers + 1

markers[missing==255] = 0

markers = cv2.watershed(img, markers)

image, cntrs, hierarchy = cv2.findContours(markers.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

for i in range(len(cntrs)):
  if hierarchy[0][i][3] == -1: cv2.drawContours(img, cntrs, i, 255, 10)

helpers.implot(img)
