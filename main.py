import cvzone
import cv2
import numpy as np
import math
import random
from cvzone.HandTrackingModule import HandDetector

# Video capture
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = HandDetector(detectionCon=0.5, maxHands=1)

class SnakeGameClass:
    def __init__(self, pathFood):
        self.points = []  # All points of the snake
        self.lengths = []  # Distance between each point
        self.currentLength = 0  # Total length of the snake
        self.allowedLength = 150  # Max allowed length
        self.previousHead = None  # Previous head position

        self.imgFood = cv2.imread(pathFood, cv2.IMREAD_UNCHANGED)
        self.imgFood = cv2.resize(self.imgFood, (80, 80))  # Resize the apple image
        self.hFood, self.wFood, _ = self.imgFood.shape
        self.foodPoint = 0, 0
        self.randomFoodLocation()

        self.score = 0
        self.gameOver = False

    def randomFoodLocation(self):
        self.foodPoint = random.randint(100, 1000), random.randint(100, 600)

    def update(self, imgMain, currentHead):
        if self.gameOver:
            cvzone.putTextRect(imgMain, "Game Over", [300, 400],
                               scale=7, thickness=5, offset=20)
            cvzone.putTextRect(imgMain, f'Your Score: {self.finalScore}', [300, 550],
                               scale=7, thickness=5, offset=20)
            return imgMain

        cx, cy = currentHead

        if self.previousHead is None:
            self.previousHead = cx, cy
            self.points.append([cx, cy])
            return imgMain

        px, py = self.previousHead
        self.points.append([cx, cy])
        distance = math.hypot(cx - px, cy - py)
        self.lengths.append(distance)
        self.currentLength += distance
        self.previousHead = cx, cy

        # Remove tail if too long
        if self.currentLength > self.allowedLength:
            for i, length in enumerate(self.lengths):
                self.currentLength -= length
                self.lengths.pop(i)
                self.points.pop(i)
                if self.currentLength < self.allowedLength:
                    break

        # Check collision with self
        if len(self.points) > 10:
            pts = np.array(self.points[:-5], np.int32)  # Ignore last few points near the head
            pts = pts.reshape((-1, 1, 2))
            minDist = cv2.pointPolygonTest(pts, (cx, cy), True)
            if minDist >= 0 and minDist < 10:
                self.gameOver = True
                self.finalScore = self.score  # Save final score
                self.points = []
                self.lengths = []
                self.currentLength = 0
                self.allowedLength = 150
                self.previousHead = None
                self.score = 0
                self.randomFoodLocation()
                return imgMain

        # Draw snake
        for i in range(1, len(self.points)):
            cv2.line(imgMain, self.points[i - 1], self.points[i], (0, 0, 255), 20)
        cv2.circle(imgMain, self.points[-1], 20, (255, 0, 255), cv2.FILLED)

        # Draw food
        rx, ry = self.foodPoint
        imgMain = cvzone.overlayPNG(imgMain, self.imgFood,
                                    (rx - self.wFood // 2, ry - self.hFood // 2))

        # Check food eaten
        if rx - self.wFood // 2 < cx < rx + self.wFood // 2 and \
           ry - self.hFood // 2 < cy < ry + self.hFood // 2:
            self.allowedLength += 50
            self.score += 1
            self.randomFoodLocation()

        # Draw Score
        cvzone.putTextRect(imgMain, f'Score: {self.score}', [50, 80],
                           scale=3, thickness=3, offset=10)

        return imgMain


# Load game with apple image
game = SnakeGameClass("Apple.png")

# Game loop
while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    hands, img = detector.findHands(img)

    if hands:
        lmList = hands[0]['lmList']
        pointIndex = lmList[8][0:2]
        img = game.update(img, pointIndex)
    else:
        # Reset game if no hands and game over
        if game.gameOver:
            game = SnakeGameClass("Apple.png")

    cv2.imshow("Snake Game", img)
    if cv2.waitKey(1) == ord('q'):
        break
