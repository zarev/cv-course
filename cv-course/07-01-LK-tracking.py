import numpy as np
import cv2
import matplotlib.pyplot as plt
import helpers
import time

corner_params = dict(maxCorners = 10, 
                     qualityLevel = 0.3, 
                     minDistance = 7, 
                     blockSize = 7)

lk_params = dict(winSize = (200,200), 
                 maxLevel = 2, 
                 criteria = (cv2.TERM_CRITERIA_EPS | cv2.TermCriteria_COUNT, 10, 0.03))

cap = cv2.VideoCapture('data/vid.mp4')

# Grab the very first frame of the stream
ret, prev_frame = cap.read()

# Grab a grayscale image (We will refer to this as the previous frame)
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

# points to track
prevPts = cv2.goodFeaturesToTrack(prev_gray, mask=None, **corner_params)

# for visualising
mask = np.zeros_like(prev_frame)

while True: 

  # Grab current frame
  ret, curr_frame = cap.read()

  frame_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

  nextPts, status, _ = cv2.calcOpticalFlowPyrLK(
    prev_gray, frame_gray, 
    prevPts, None, 
    **lk_params)
    
  # Using the returned status array (the status output)
  # status output status vector (of unsigned chars); each element of the vector is set to 1 if
  # the flow for the corresponding features has been found, otherwise, it is set to 0.
  good_new = nextPts[status==1]
  good_prev = prevPts[status==1]

  for i, (new,prev) in enumerate(zip(good_new, good_prev)):
    # flattens out array
    x_new, y_new = new.ravel()
    x_prev, y_prev = prev.ravel()
 
    mask = cv2.line(mask, (x_new, y_new), (x_prev, y_prev), (0,255,0,), 3)

    curr_frame = cv2.circle(curr_frame, (x_new, y_new), 4, (0,0,255), -1)

  img = cv2.add(curr_frame, mask)
  cv2.imshow('tracking', img)

  k = cv2.waitKey(30) & 0xff
  if k == 27:
      break

  prev_gray = frame_gray.copy()
  prevPts = good_new.reshape(-1, 1, 2)


cv2.destroyAllWindows()
cap.release()


  
