import cv2
import mediapipe as mp
import time
import AIHandTrackingModule as atm


pTime = 0
cTime = 0
cap = cv2.VideoCapture(0)
tracker = atm.handTracker()  # calling the tracker class
while True:
    success, img = cap.read()
    img = tracker.findHands(img)  # calling the method "find hands" within our class tracker
    pointList = tracker.findPosition(img)
    if len(pointList) != 0:
        print(pointList[4])

    cTime = time.time()  # current time
    fps = 1 / (cTime - pTime)  # currentime minus present time for the frames
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 0, 255), 2)

    cv2.imshow("Image", img)
    cv2.waitKey(1)