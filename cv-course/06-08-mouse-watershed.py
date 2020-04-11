import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import cm
import helpers

road = helpers.imread('data/road_image.jpg')
road_cp = np.copy(road)

marker_image = np.zeros(road.shape[:2], dtype=np.int32)

segments = np.zeros(road.shape, dtype=np.uint8)

def creare_rgb(i):
  cmap = cm.get_cmap("tab10")
  return tuple(np.array(cmap(i)[:3])*255)

colors = []

for i in range(10): colors.append(creare_rgb(i))

n_markers = 10
curr_marker = 1
marks_updated = False
# cv2.circle(road_cp, (10,10), 10, colors[curr_marker, -1])


# handles mouse actions
def mouse_callback(event, x, y, flags, param):
  global marks_updated
  while 1:
    if event == cv2.EVENT_LBUTTONDOWN:
      # markers passed to the watershed algorithm
      cv2.circle(marker_image, (x,y), 10, (curr_marker), -1)

      # user sees on the road image
      cv2.circle(road_cp, (x,y), 10, colors[curr_marker], -1)

      marks_updated = True
    break

road_window = 'Road Image'
cv2.namedWindow(road_window)
cv2.setMouseCallback(road_window, mouse_callback)

while 1:

  cv2.imshow('Watershed Segments', segments)
  cv2.imshow(road_window, road_cp)

  # show current color in top left corner
  cv2.circle(road_cp, (10,10), 10, colors[curr_marker], -1)

  # if press ESC close windows
  k = cv2.waitKey(1)
  if k == 27: break

  # clear colours if pressed 'C'
  elif k==ord('c'): 
    road_cp = road.copy()
    marker_image = np.zeros(road.shape[:2], dtype=np.int32)
    segments = np.zeros(road.shape, dtype = np.uint8)
  
  # update color choice
  elif k > 0 and chr(k).isdigit(): 
    curr_marker = int(chr(k))

  
  # if left button, update markings
  if marks_updated:
    marker_image_cp = marker_image.copy()
    cv2.watershed(road, marker_image_cp)

    segments = np.zeros(road.shape, dtype=np.uint8)


    # coloring segments
    for color_ind in range(n_markers): 
      segments[marker_image_cp==(color_ind)] = colors[color_ind]

cv2.destroyAllWindows()