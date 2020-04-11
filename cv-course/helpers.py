import numpy as np
import cv2
import matplotlib.pyplot as plt

# plotting for non notebooks
# no need for cmap if imread below
def implot(img, cmap = None):
  plt.figure(figsize=(12,10))
  plt.imshow(img, cmap)
  plt.show()

# imports image from source
# handles color automatically,
# based on standard imread input
def imread(source, hasColor = 0):
  img = cv2.imread(source, hasColor)
  if hasColor: return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  if hasColor == 2 : return img
  return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)