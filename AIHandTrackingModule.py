import cv2
import mediapipe as mp
import time

class handTracker():
    def __init__(self, mode = False, maxHands = 2, detectCon = 0.5, trackCon = 0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectCon = detectCon
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.detectCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils  # method provided in mediapipe to draw the points on hands

    def findHands(self,img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # it just takes RGB images only
        self.results = self.hands.process(imgRGB)  #
        #print(results.multi_hand_landmarks)
        if self.results.multi_hand_landmarks:
            for handPoints in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handPoints, self.mpHands.HAND_CONNECTIONS) #orignal image for single hand
        return img

    def findPosition(self,img, handNo=0, draw = True): # we are doing it on for one hand but we can change the nad no if we want it for more hands

        pointList = []
        if self.results.multi_hand_landmarks:
            OneHand = self.results.multi_hand_landmarks[handNo] #getting first hand

            for id, points in enumerate(OneHand.landmark): #within first hand getting all the points
                h, w, c = img.shape  # height, width and channels of img
                cx, cy = int(points.x * w), int(points.y * h)  # position of center
                #print(id, cx, cy)  # we are printing id along with cx, cy to know the points with id nos.
                #if id == 0: #drawing for each landmark/points
                pointList.append([id,cx,cy]) #return the list of the points
                if draw:
                    cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
        return pointList


def main():                  #dummy code to showcase what can this module do
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    tracker = handTracker() #calling the tracker class
    while True:
        success, img = cap.read()
        img = tracker.findHands(img) #calling the method "find hands" within our class tracker
        pointList = tracker.findPosition(img)
        if len(pointList) != 0:
            print(pointList[4])

        cTime = time.time()  # current time
        fps = 1 / (cTime - pTime)  # currentime minus present time for the frames
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 0, 255), 2)

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":  # if we are running this script , do this. whatever we write in the dummy code
    main()