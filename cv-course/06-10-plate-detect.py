import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import cm
import helpers

img = helpers.imread('data/car_plate.jpg')

plate_cas = cv2.CascadeClassifier('data/haarcascades/haarcascade_russian_plate_number.xml')


def detect_plate(img):
  img = img.copy()
  
  plate_recs = plate_cas.detectMultiScale(img, scaleFactor=1.2, minNeighbors=5)
  for (x,y,w,h) in plate_recs:
    cv2.rectangle(img, (x,y), (x+w,y+h), (255), 1)
    # separate plate to blur
    plate_rec = img[y:y+h,x:x+w]
    # blur plate
    plate_rec = cv2.medianBlur(plate_rec, 9)
    # replace plate with blurred plate
    img[y:y+h,x:x+w] = plate_rec

  return img
  
helpers.implot(detect_plate(img))
