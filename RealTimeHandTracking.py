import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands(  )
mp.Draw = mp.solutions.drawing_utils #method provided in mediapipe to draw the points on hands

pTime = 0
cTime = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # it just takes RGB images only
    results = hands.process(imgRGB)  #
    #print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handPoints in results.multi_hand_landmarks:
            for id,points in enumerate(handPoints.landmark):
                #print(id,points)
                h , w, c = img.shape #height, width and channels of img
                cx , cy = int(points.x*w), int(points.y*h) #position of center
                print(id, cx, cy) #we are printing id along with cx, cy to know the landmarks with id nos.
                if id == 0: #drawing for each landmark/points
                    cv2.circle(img,(cx,cy),20 ,(255,0,255),cv2.FILLED)


            mp.Draw.draw_landmarks(img, handPoints, mpHands.HAND_CONNECTIONS) #orignal image for single hand



    cTime = time.time() # current time
    fps = 1/(cTime-pTime) #currentime minus present time
    pTime = cTime

    cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_COMPLEX,3,(255,0,255),2)


    cv2.imshow("Image",img)
    cv2.waitKey(1)