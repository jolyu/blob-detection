import cv2
import numpy as np
from . import filters
from . import image_operations as img_o
from logging_framework import logging_setup as log


#------------------------------   v GLOBAL VARIABLES v   ------------------------------

#blob detect parameters:
MAX_THRESHOLD = 93
MIN_THRESHOLD = 65
MAX_AREA = 1000
MIN_AREA = 10
MAX_CIRCULARITY = 1
MIN_CIRCULATIRY = 0.4
MAX_CONVEXITY = 1
MIN_CONVEXITY = 0.0
MAX_INETRTIA_RATIO = 1
MIN_INERTIA_RATIO = 0.0


DEBUG = False #uncomment to use imshow etc.

#------------------------------   ^GLOBAL VARIABLES^   ------------------------------

#------------------------------   v FUNCTIONS v   ------------------------------

def init_blob_detector():
    # Setup SimpleBlobDetector parameters.
    params = cv2.SimpleBlobDetector_Params()

    # Change thresholds
    #params.maxThreshold = MAX_THRESHOLD
    #params.minThreshold = MIN_THRESHOLD

    #params.filterByColor = False
    #params.blobColor = False
    
    # Filter by Area.
    params.filterByArea = False
    #params.maxArea = MAX_AREA
    #params.minArea = MIN_AREA
 

    # Filter by Circularity
    params.filterByCircularity = False
    #params.maxCircularity = MAX_CIRCULARITY
    #params.minCircularity = MIN_CIRCULATIRY

    # Filter by Convexity
    params.filterByConvexity = False
    #params.minConvexity = MIN_CONVEXITY
    #params.maxConvexity = MAX_CONVEXITY

    # Filter by Inertia
    params.filterByInertia = False
    #params.maxInertiaRatio = MAX_INERTIA_RATIO
    #params.minInertiaRatio = MIN_INERTIA_RATIO

    detector = cv2.SimpleBlobDetector_create(params)    #create detector

    return detector
    
def draw_blobs(img, keyPoints):     
    imgKeyPoints = cv2.drawKeypoints(img, keyPoints, np.array([]),(0,0,255), cv2.DRAW_MATCHES) #DRAW_MATCHES_FLAGS_RICH KEYPOINTS for cicles and not dots
    cv2.imshow('blobs', imgKeyPoints)                                                          #show img

def blob_detection(img):
    #function to detect blobs. Returns list of keypoints

    detector = init_blob_detector() #create detector with params for blobs

    keyPoints = detector.detect(img) #analyse and make list of blobs in picture

    if DEBUG:
        draw_blobs(img, keyPoints)  #show img with red cicles if debug
    
    return keyPoints

#------------------------------   ^FUNCTIONS^   ------------------------------

#------------------------------   v TEST FUNCTION v   ------------------------------

def blob_detection_test_func():
    #just some function to test
    #currently empty
    pass

#------------------------------   ^TEST FUNCTION^   ------------------------------

if __name__ == "__main__":
    pass