import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO

cam = cv2.VideoCapture(0)
verif = True

model = YOLO("runs\\detect\\train-8\\weights\\best.pt")

if not cam.isOpened():
    print("Cam pas ouverte")
    verif = False

while verif:
    ret, frame = cam.read()
    #ret vérifie si frame existe et frame c'est l'image
    frame_miroir = cv2.flip(frame, 1)

    
    results = model(frame_miroir, stream=True)

    for r in results:
        annoted_frame = r.plot()

    cv2.imshow('frame', annoted_frame)

    if cv2.waitKey(1) == ord('q'):
        break

cam.release() 
cv2.destroyAllWindows()