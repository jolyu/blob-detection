import pafy
import youtube_dl
import cv2
import blobDetect as bd 

url2 = 'https://www.youtube.com/watch?v=a1wp1RnC7kk'
url = 'https://www.youtube.com/watch?v=QoaMc9eYxCg'

vPafy = pafy.new(url)
play = vPafy.getbest(preftype="webm")

#start the video
cap = cv2.VideoCapture(play.url)
ret,frame = cap.read()

cv2.imshow("preview", frame)
detector = bd.blobDetector()

while (True):
    ret,frame = cap.read()
    
    newFrame = bd.detectStuff(frame, detector)
    cv2.imshow("preview", newFrame)

    cv2.imshow('frame',newFrame)
    if cv2.waitKey(30) == 27:
        break    

cap.release()
cv2.destroyAllWindows()