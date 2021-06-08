import cv2
from joblib import load
import numpy as np
from skimage.feature import hog
import model as clf
# clf = load('filename.joblib') 
# Load the cascade
face_cascade = cv2.CascadeClassifier('base/haarcascade_frontalface_default.xml')

# To capture video from webcam. 
cap = cv2.VideoCapture(0)
# To use a video file as input 
# cap = cv2.VideoCapture('filename.mp4')

while True:
    # Read the frame
    _, img = cap.read()

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect the faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Draw the rectangle around each face
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        crop_img = img[y:y+h, x:x+w]

    # Display
    cv2.imshow('img', img)
    try:
        cv2.imshow("cropped", crop_img)
        resized = cv2.resize(img, (80,80), interpolation = cv2.INTER_AREA)
        print(model.classify(resized))
    except:
        pass
    
    print(('happy','sad')[clf.predict(resized)])
    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k==27:
        break
        
# Release the VideoCapture object
cap.release()