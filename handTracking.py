import cv2
import mediapipe as mp
import time


class handDetector():
    def __init__(self,mode=False,maxHands=2,detectionConfidence=0.5,trackConfidence=0.5,modelC=1):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionConfidence = detectionConfidence
        self.trackConfidence = trackConfidence
        self.mediaHands = mp.solutions.hands
        self.modelC = modelC
        self.hands = self.mediaHands.Hands(self.mode,self.maxHands,self.modelC,self.detectionConfidence,self.trackConfidence) #go with default parameters
        self.mediaDraw = mp.solutions.drawing_utils


    def findHands(self,img,draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for eachHandLM in self.results.multi_hand_landmarks:
                if draw:
                    self.mediaDraw.draw_landmarks(img, eachHandLM, self.mediaHands.HAND_CONNECTIONS)

        return img

    def findPosition(self, img , handNum=0 , draw = True):
        landmarkList =[]
        if  self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNum]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                landmarkList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (100, 0, 120), cv2.FILLED)

        return landmarkList


def main():
    cam = cv2.VideoCapture(0)

    previousTime = 0
    currentTime = 0
    detector = handDetector()
    while True:
        success, img = cam.read()

        img = detector.findHands(img)
        landmarkList = detector.findPosition(img)

        if(len(landmarkList)!=0):
            print(landmarkList[4])

        currentTime = time.time()
        fps = 1 / (currentTime - previousTime)
        previousTime = currentTime
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()
