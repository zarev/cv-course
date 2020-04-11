import numpy as np
import cv2
import matplotlib.pyplot as plt
import helpers
import time

def detect_face(img):
  # img = img.copy()

  face_cas = cv2.CascadeClassifier('data/haarcascades/haarcascade_frontalface_default.xml')

  face_recs = face_cas.detectMultiScale(img, scaleFactor=1.2 , minNeighbors=9)

  for (x,y,w,h) in face_recs:
    # cv2.rectangle(img, (x,y), (x+w,y+h), (255), 1)
    # separate face to blur
    # face_rec = img[y:y+h,x:x+w]
    # blur face
    # face_rec = cv2.medianBlur(face_rec, 19)
    # replace face with blurred face
    # img[y:y+h,x:x+w] = face_rec
  
    return [x,y,x+w,y+h]

tracker = cv2.TrackerTLD_create()
tracker_name = str(tracker).split()[0][1:]

# Read video
cap = cv2.VideoCapture(0)

# Read first frame.
ret, frame = cap.read()

# Special function allows us to draw on the very first frame our desired ROI
roi = cv2.selectROI(frame, False)
roi2 = tuple(detect_face(frame))
print(roi)
print(roi2)
# roi = tuple(detect_face(frame))

# Initialize tracker with first frame and bounding box
ret = tracker.init(frame, roi)

print(type(roi))

while True:
    # Read a new frame
    ret, frame = cap.read()
    
    
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
        cv2.putText(frame, "Cannot see a face :(", (100,200), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),3)

    # Display tracker type on frame
    cv2.putText(frame, tracker_name, (20,400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),3)

    # Display result
    cv2.imshow(tracker_name, frame)

    # Exit if ESC pressed
    k = cv2.waitKey(1) & 0xff
    if k == 27 : 
        break
        
cap.release()
cv2.destroyAllWindows()