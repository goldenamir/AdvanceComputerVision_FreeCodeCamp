# Importing required libraries
import cv2
import mediapipe as mp
import time

# Opening the camera
cap = cv2.VideoCapture(2)  # number one goes for goPro

# making an instance (called mphands) from meidapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0  # previous time is equal with zero
cTime = 0  # current time is equal with zero

while True:
    # while loop for reading each frame
    success, img = cap.read()

    # chaning to RGB
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # checking the result of imgRGB in hand
    results = hands.process(imgRGB)
    # check the results has any information
    print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                # print(id, lm) # it gives us the id and landmarks for a hands
                h, w, c = img.shape  # gives us height, width and channel of the Image
                cx, cy = int(lm.x*w), int(lm.y*h)
                print(id, cx, cy)  # id, coordination of landmarks

                # drawing a circle in the specific landmark
                if id == 0:  # drawing a circle on the wrist
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

                if id == 4:  # drawing a circle on the thumb
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 70),
                cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    cv2.imshow('Image', img)
    cv2.waitKey(1)
