import pandas as pd
import numpy
import cv2
import numpy as np
import sys

# cascPath = sys.argv[1]
faceCascade = cv2.CascadeClassifier('C:\\Users\\91866\\PycharmProjects\\mask detector\\venv\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml')
# faceCascade = cv2.CascadeClassifier()
eyeCascade = cv2.CascadeClassifier('C:\\Users\\91866\\PycharmProjects\\mask detector\\venv\\Lib\\site-packages\\cv2\\data\\haarcascade_eye.xml')
smileCascade = cv2.CascadeClassifier('C:\\Users\\91866\\PycharmProjects\\mask detector\\venv\\Lib\\site-packages\\cv2\\data\\haarcascade_smile.xml')

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, img = cap.read()
    faces = faceCascade.detectMultiScale(
        img,
        scaleFactor=1.5,
        minNeighbors=5,
        minSize=(30, 30)
    )
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        # roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]

        eyes = eyeCascade.detectMultiScale(
            roi_color,
            scaleFactor=1.5,
            minNeighbors=5,
            minSize=(5, 5),
        )

        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

        smile = smileCascade.detectMultiScale(
            roi_color,
            scaleFactor=1.5,
            minNeighbors=15,
            minSize=(25, 25),
        )

        for (xx, yy, ww, hh) in smile:
            cv2.rectangle(roi_color, (xx, yy), (xx + ww, yy + hh), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('frame',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

