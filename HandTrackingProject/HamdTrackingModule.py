# Importing required libraries
import cv2
import mediapipe as mp
import time


class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionConfidence=0.5, trackConfidence=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionConfidence = detectionConfidence
        self.trackConfidence = trackConfidence

        # making an instance (called mphands) from meidapipe
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            self.mode, self.maxHands, self.detectionConfidence, self.trackConfidence)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img):
        
        # chaning to RGB
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # checking the result of imgRGB in hand
        results = self.hands.process(imgRGB)
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

                self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)


def main():

    pTime = 0  # previous time is equal with zero
    cTime = 0  # current time is equal with zero
    # Opening the camera
    cap = cv2.VideoCapture(2)  # number one goes for goPro

    while True:
        # while loop for reading each frame
        success, img = cap.read()

        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70),
                    cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        cv2.imshow('Image', img)
        cv2.waitKey(1)


if __name__ == '__main__':
    main()
