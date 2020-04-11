import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import cm
import helpers

nadia = helpers.imread('data/Nadia_Murad.jpg')
denis = helpers.imread('data/Denis_Mukwege.jpg')
solvay = helpers.imread('data/solvay_conference.jpg')

face_cas = cv2.CascadeClassifier('data/haarcascades/haarcascade_frontalface_default.xml')
eyes_cas = cv2.CascadeClassifier('data/haarcascades/haarcascade_eye.xml')

def adj_detect_face(img):
  face = img.copy()
  
  face_recs = face_cas.detectMultiScale(face, scaleFactor=1.2, minNeighbors=5)

  for (x,y,w,h) in face_recs:
    cv2.rectangle(face, (x,y), (x+w,y+h), (255), 1)
  
  return face

def detect_eyes(img):
  face = img.copy()
  
  eye_recs = eyes_cas.detectMultiScale(face)

  for (x,y,w,h) in eye_recs:
    cv2.rectangle(face, (x,y), (x+w,y+h), (255), 1)
  
  return face

cap = cv2.VideoCapture('data/video_capture.mp4')

while 1:
  ret,frame = cap.read(0)
  frame = adj_detect_face(frame)
  helpers.implot(frame)

  k = cv2.waitKey(1)
  if k==27: break
cap.release()
cv2.destroyAllWindows()


# helpers.implot(detect_eqqqqqqqqqyes(nadia))
