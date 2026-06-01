import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math

# 1. Configuration du détecteur
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

# Note : Tu auras besoin du fichier 'hand_landmarker.task' 
# Il se télécharge automatiquement ou via : 
# https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task

cap = cv2.VideoCapture(0)

compteur = 0
while cap.isOpened():
    ret, frame = cap.read()
    frame = cv2.resize(frame, (1480, 1080))
    if not ret: break

    # Conversion en image MediaPipe
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

    # Détection
    detection_result = detector.detect(mp_image)

    if detection_result.hand_landmarks:
        for hand_landmarks in detection_result.hand_landmarks:
            for landmark in hand_landmarks:
                x1 = int(landmark.x * frame.shape[1])
                y1 = int(landmark.y * frame.shape[0])

                cv2.circle(frame, (x1, y1), 5, (0, 255, 0), -1)

            p1 = (hand_landmarks[4].x, hand_landmarks[4].y)
            p2 = (hand_landmarks[8].x, hand_landmarks[8].y)


            p1_pri = (int(hand_landmarks[4].x*frame.shape[1]), int(hand_landmarks[4].y*frame.shape[0]))
            p2_pri = (int(hand_landmarks[8].x*frame.shape[1]), int(hand_landmarks[8].y*frame.shape[0]))

            cv2.line(frame, p1_pri, p2_pri, (255, 0, 0), 2)

            distance = math.dist(p1, p2)
            if distance < 0.02:
                compteur += 1
                
    frame = cv2.flip(frame, 1)

    cv2.putText(frame, f"Compteur : {compteur}", (0,45), 2, 2, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('Hand Tracking Tasks', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()