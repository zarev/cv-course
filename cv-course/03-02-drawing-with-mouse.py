import numpy as np
import cv2
import matplotlib.pyplot as plt

def draw_cirle(event, x, y, flags, param): 
  if event == cv2.EVENT_LBUTTONDOWN:
    cv2.circle(img, (x, y), 10, (255,0,0), -1)
  elif event == cv2.EVENT_RBUTTONDOWN: 
    cv2.circle(img, (x, y), 10, (0,255,0), -1)
  

cv2.namedWindow(winname="cover")
cv2.setMouseCallback("cover", draw_cirle)


img = cv2.imread('cover.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

cv2.rectangle(img, pt1=(100,100), pt2=(300,300), color=(0,255,0), thickness=1)

cv2.circle(img, center=(100,100), radius=10, color = (0,255,0), thickness=-1)
cv2.circle(img, center=(300,300), radius=10, color = (0,255,0), thickness=-1)
cv2.circle(img, center=(300,100), radius=10, color = (0,255,0), thickness=-1)
cv2.circle(img, center=(100,300), radius=10, color = (0,255,0), thickness=-1)

font = cv2.FONT_HERSHEY_COMPLEX
cv2.putText(img, text = 'Only one survives.', org=(170,450), fontFace=font, fontScale=1, color=(255,255,255))

cv2.line(img, (260, 460), (320, 460), (255,255,255), 1, cv2.LINE_AA)

while 1:
  cv2.imshow("cover", img)
  if cv2.waitKey(20) & (0xFF==27 or cv2.getWindowProperty("cover", 0) == -1 ): break

cv2.destroyAllWindows()
