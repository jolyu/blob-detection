import cv2
import numpy as np
from . import filters
from . import image_operations as img_o


def initBlobDetector():
    # Setup SimpleBlobDetector parameters.
    params = cv2.SimpleBlobDetector_Params()

    # Change thresholds
    #params.minThreshold = 65
    #params.maxThreshold = 93

    params.filterByColor = False
    #params.blobColor = False
    # Filter by Area.
    params.filterByArea = False
    params.minArea = 10
    #params.maxArea = 5000

    # Filter by Circularity
    params.filterByCircularity = False
    #params.minCircularity = 0.4
    #params.maxCircularity = 1

    # Filter by Convexity
    params.filterByConvexity = False
    #params.minConvexity = 0.0

    # Filter by Inertia
    params.filterByInertia = False
    params.minInertiaRatio = 0.0

    detector = cv2.SimpleBlobDetector_create(params)

    return detector

def blobDetection(img):
    #function to detect blobs. Returns list of keypoints

    detector = initBlobDetector()

    keyPoints = detector.detect(img)

    #draw detected keypoints as red circles
    #imgKeyPoints = cv2.drawKeypoints(img, keyPoints, np.array([]),(0,0,255), cv2.DRAW_MATCHES_FLAGS_DEFAULT) #uncomment for testing
    #cv2.imshow('blobs', imgKeyPoints)
    return keyPoints

def blob_detection_test_func():
    print('Lets have a look at all cases of filter func')
    img = img_o.readImage('FLIR0023.jpg')
    cv2.imshow('org', img)

    img1 = filters.filterImg(img, 0, True)
    cv2.imshow('0M', img1)

    img2 = filters.filterImg(img, 1, True)
    cv2.imshow('1M', img2)

    img3 = filters.filterImg(img, 2, True)
    cv2.imshow('2M', img3)

    img4 = filters.filterImg(img, 0, False)
    cv2.imshow('0', img4)

    img5 = filters.filterImg(img, 1, False)
    cv2.imshow('1', img5)

    img6 = filters.filterImg(img, 2, False)
    cv2.imshow('2', img6)

    while True:
        if cv2.waitKey(30) == 27: # exit on ESC                     #avslutter programmet og lukker alle viduer dersom man trykker ESC
            break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    blob_detection_test_func()