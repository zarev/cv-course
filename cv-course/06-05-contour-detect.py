import numpy as np
import cv2
import matplotlib.pyplot as plt
import helpers

img = cv2.imread('resources/internal_external.png', 0)

helpers.implot(img)

image, cntrs, hierarchy = cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

ext_cons = np.zeros(image.shape)

for i in range(len(cntrs)):
  if(hierarchy[0][i][3]) == -1: cv2.drawContours(ext_cons, cntrs, i, 255, -1)

helpers.implot(ext_cons)

int_cons = np.zeros(image.shape)

for i in range(len(cntrs)):
  if (hierarchy[0][i][3]) != -1: cv2.drawContours(int_cons, cntrs, i, 255, -1)

helpers.implot(int_cons)
