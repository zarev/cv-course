import numpy as np
import cv2
import matplotlib.pyplot as plt
import helpers
import time

def detect_face(img):
  # img = img.copy()
  
  face_recs = face_cas.detectMultiScale(img, scaleFactor=1.02 , minNeighbors=9)

  for (x,y,w,h) in face_recs:
    # cv2.rectangle(img, (x,y), (x+w,y+h), (255), 1)
    # separate face to blur
    face_rec = img[y:y+h,x:x+w]
    # blur face
    face_rec = cv2.medianBlur(face_rec, 19)
    # replace face with blurred face
    img[y:y+h,x:x+w] = face_rec
  
  return img


cap = cv2.VideoCapture(0)

face_cas = cv2.CascadeClassifier('data/haarcascades/haarcascade_frontalface_default.xml')

cv2.namedWindow('frame')

while cap.isOpened():
  ret, frame = cap.read(0)

  detect_face(frame)

  if ret == True:
    # time.sleep(1/60)
    cv2.imshow('frame', frame)  
    if cv2.waitKey(1) & 0xFF == ord('q'): break
  else: break
  
cap.release()
cv2.destroyAllWindows()