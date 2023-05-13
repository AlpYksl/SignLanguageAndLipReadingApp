import cv2
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import os

cap = cv2.VideoCapture("videos/bitte.mp4")
cap.set(3,648)
cap.set(4,488)
cap.set(cv2.CAP_PROP_FPS,60)
segmentor = SelfiSegmentation()
fpsReader = cvzone.FPS()
listImgs = os.listdir("Images")
print(listImgs)
imgBG = cv2.imread("Batman.jpg")
while True:
    success, img = cap.read()
    imgOut = segmentor.removeBG(img,(0,0,255), threshold= 0.3)
    _, imgOut = fpsReader.update(imgOut, color=(0,0,255))
    cv2.imshow('Image', imgOut)
    
    cv2.waitKey(1)
