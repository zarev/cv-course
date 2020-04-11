import numpy as np
import cv2
import matplotlib.pyplot as plt
import helpers as hlp

img = hlp.imread('data/sammy_face.jpg')

med_val = np.median(img)

blur = cv2.blur(img, (4,3))

edges = cv2.Canny(blur, 127, 255)

hlp.implot(edges)