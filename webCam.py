import cv2
import blobDetect as bd


if __name__ == "__main__":
    cv2.namedWindow("preview")
    vc = cv2.VideoCapture(0)

    if vc.isOpened(): # try to get the first frame
        rval, frame = vc.read()
    else:
        rval = False

    cv2.imshow("preview", frame)
    detector = bd.blobDetector()

    while rval:
    
        rval, frame = vc.read()

        newFrame = bd.detectStuff(frame, detector)
        cv2.imshow("preview", newFrame)

    
        if cv2.waitKey(30) == 27: # exit on ESC
            break
    cv2.destroyWindow("preview")
