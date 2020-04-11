import numpy as np
import cv2
import matplotlib.pyplot as plt
import helpers
import time

def blur_face(img):
  # img = img.copy()

  tracker = tracker = cv2.TrackerTLD_create()

  face_recs = face_cas.detectMultiScale(img, scaleFactor=1.2 , minNeighbors=9)

  for (x,y,w,h) in face_recs:
    # cv2.rectangle(img, (x,y), (x+w,y+h), (255), 1)
    # separate face to blur
    face_rec = img[y:y+h,x:x+w]
    # blur face
    # face_rec = cv2.medianBlur(face_rec, 19)
    # replace face with blurred face
    img[y:y+h,x:x+w] = face_rec
  
  return face_rec

tracker = cv2.TrackerTLD_create()

cap = cv2.VideoCapture('data/vid.mp4')

face_cas = cv2.CascadeClassifier('data/haarcascades/haarcascade_frontalface_default.xml')

ret, frame = cap.read()


while 1:
# Update tracker
  success, roi = tracker.update(frame)
  
  # roi variable is a tuple of 4 floats
  # We need each value and we need them as integers
  (x,y,w,h) = tuple(map(int,roi))
  
  # Draw Rectangle as Tracker moves
  if success:
      # Tracking success
      p1 = (x, y)
      p2 = (x+w, y+h)
      cv2.rectangle(frame, p1, p2, (0,255,0), 3)
  else :
      # Tracking failure
      cv2.putText(frame, "Failure to Detect Tracking!!", (100,200), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),3)

  # Display result
  cv2.imshow('frame', frame)

  if ret == True:
    # time.sleep(1/60)
    if cv2.waitKey(1) & 0xFF == 27: break
  else: break
  
cap.release()
cv2.destroyAllWindows()