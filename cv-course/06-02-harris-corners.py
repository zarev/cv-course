import numpy as np
import cv2
import matplotlib.pyplot as plt
import helpers as hlp

flat_chess = hlp.imread('data/flat_chessboard.png')
flat_gray = cv2.cvtColor(flat_chess, cv2.COLOR_BGR2GRAY)

# hlp.implot(flat_gray, cmap='gray')

real_chess = hlp.imread('data/real_chessboard.jpg')
real_gray = cv2.cvtColor(real_chess, cv2.COLOR_BGR2GRAY)

# hlp.implot(real_gray, cmap='gray')

gray = np.float32(flat_gray)
dst = cv2.cornerHarris(gray, 2, 3, 0.04)
dst = cv2.dilate(dst, None)

flat_chess[dst>0.01*dst.max()] = [255, 0, 0]
hlp.implot(flat_chess)

gray = np.float32(real_gray)
dst = cv2.cornerHarris(gray, 2, 3, 0.04)
dst = cv2.dilate(dst, None)

real_chess[dst>0.01*dst.max()] = [255, 0, 0]
hlp.implot(real_chess)

# shi-tomasi
corners = cv2.goodFeaturesToTrack(image=flat_gray, maxCorners=5, qualityLevel=0.01,minDistance=10)
corners = np.int0(corners)

for i in corners:
  x, y = i.ravel()
  cv2.circle(flat_chess, (x,y), 2, (255, 0, 0), -1)
  
hlp.implot(flat_chess)


corners = cv2.goodFeaturesToTrack(image=real_gray, maxCorners=5, qualityLevel=0.01,minDistance=10)
corners = np.int0(corners)

for i in corners:
  x, y = i.ravel()
  cv2.circle(real_chess, (x,y), 2, (255, 0, 0), -1)
  
hlp.implot(real_chess)



