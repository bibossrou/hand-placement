import cv2
import numpy as np
import mediapipe as mp

cam = cv2.VideoCapture(0)
verif = True

if not cam.isOpened():
    print("Cam pas ouverte")
    verif = False

while verif:
    ret, frame = cam.read()
    #ret vérifie si frame existe et frame c'est l'image
    frame_miroir = cv2.flip(frame, 1)
    cv2.imshow('frame', frame_miroir)

    if cv2.waitKey(1) == ord('q'):
        break


cam.release() 
cv2.destroyAllWindows()