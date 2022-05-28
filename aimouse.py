import cv2
import numpy as np
import HandTrackingModule as htm
import time
import autopy
import handDetection as hd


wCam, hCam = 640, 480
frameR = 100  # frame reduction
smoothing = 5



pTime = 0
plocx, plocy = 0, 0
clocx, clocy = 0, 0

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
# detector = htm.handDetector(maxHands=1) error in the file of module because it doesn;t contain it
detector = hd.hand_tracking_module.handDetector(maxHands=1)
wScr, hScr = autopy.screen.size()


while True:
    # 1. Find hand Landmarks
    success, img = cap.read()
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)

    # 2. Get the tip of the index and middle fingers
    if len(lmList)!=0:
        x1, y1 =  lmList[8][1:]
        x2, y2 = lmList[12][1:]


        # 3. Check which fingers are up
        fingers = detector.fingersUp()
        print(fingers)
        # 4. Only Index Finger : Moving Mode
        cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR), (255, 0, 255), 2)
        if fingers[1] == 1 and fingers[2] == 0:
            # 5. Convert Coordinates

            x3 = np.interp(x1, (frameR, wCam), (0, wScr))
            y3 = np.interp(y1, (frameR, hCam), (0, hScr))


            # 6. Smooth Value
            clocx = plocx + (x3 - plocx) / smoothing
            clocy = plocy + (y3 - plocy) / smoothing

            # 7. Move Mouse
            autopy.mouse.move(wScr-clocx, clocy)
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            plocx, plocy = clocx, clocy
        # 8. Both Index and middle fingers are up : Clicking Mode
        if fingers[1] == 1 and fingers[2] == 1:
            # 9. finding distance between fingers
            length, img, lineinfo = detector.findDistance(8, 12, img)
            print(length)
            if length < 29:
                cv2.circle(img, (x1, y1), 15, (0, 255, 0), cv2.FILLED)
                # 10. click mouse if distance is short
                autopy.mouse.click()

    # 11. Frame Rate
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                (255, 0, 0), 3)
    # 12. Display
    cv2.imshow("Image", img)
    cv2.waitKey(1)
