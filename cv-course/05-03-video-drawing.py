import numpy as np
import cv2
import matplotlib.pyplot as plt
import time


def draw_rect(event, x, y, flags, param):

  global pt1, pt2, topLeft_clicked, botRight_clicked

  if event == cv2.EVENT_LBUTTONDOWN:
    if topLeft_clicked and botRight_clicked: 
      pt1 = (0,0)
      pt2 = (0,0)
      topLeft_clicked = False
      botRight_clicked = False
    
    if topLeft_clicked == False:
      pt1 = (x, y)
      topLeft_clicked = True

    elif botRight_clicked == False:
      pt2 = (x, y)
      botRight_clicked = True 

cap = cv2.VideoCapture('data/finger_move.mp4')

pt1 = (0,0)
pt2 = (0,0)

topLeft_clicked = False
botRight_clicked = False

cv2.namedWindow('frame')
cv2.setMouseCallback('frame', draw_rect)

if cap.isOpened() == False:
  print("Error opening file.") 

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


while cap.isOpened():
  ret, frame = cap.read()

  if topLeft_clicked: cv2.circle(frame, center=pt1, radius=3, color = (0,0,255), thickness=-1)
  if topLeft_clicked and botRight_clicked:
    cv2.rectangle(frame, pt1, pt2, (0,0,255), 1)

  if ret == True:
    time.sleep(1/20)
    cv2.imshow('frame', frame)  
    if cv2.waitKey(10) & 0xFF == ord('q'): break
  else: break
  
cap.release()
cv2.destroyAllWindows()