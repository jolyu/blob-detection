import cv2
import numpy as np
from . import image_operations as img_o
from logging_framework import logging_setup as log

#------------------------------   v GLOBAL VARIABLES v   ------------------------------
SIMPLE_THRESHOLD_FILTER = 0
CV_OTZU_FILTER = 1
MANUAL_OTZU_FILTER = 2

MORPHOLOGY_ON = True
MORPHOLOGY_OFF = False

KERNEL_SIZE = 3
MIN = 0
MAX = 255
THRESHOLD_BINARY = 60

#------------------------------   ^GLOBAL VARIABLES^   ------------------------------

#------------------------------   v FUNCTIONS v   ------------------------------

def check_2D(img):
    # check if input image is in grayscale (2D)
    try:
        if img.shape[2]:
            # if there is 3rd dimension
            log.warning('otsu_binary(img) input image should be in grayscale!')
    except IndexError:
        pass  # image doesn't have 3rd dimension - proceed

def manual_otsu_binary(img):
    check_2D(img)  #Otsu binarization function by calculating threshold
    blur = cv2.GaussianBlur(img, (KERNEL_SIZE, KERNEL_SIZE), 0)   #gausian blur

    # find normalized_histogram, and its cumulative distribution function
    hist = cv2.calcHist([blur], [0], None, [256], [0, 256])
    hist_norm = hist.ravel() / hist.max()
    Q = hist_norm.cumsum()
    bins = np.arange(256)
    fn_min = np.inf
    thresh = -1

    for i in range(1, 255):
        p1, p2 = np.hsplit(hist_norm, [i])  # probabilities
        q1 = Q[i]
        q2 = Q[255] - q1  # cum sum of classes
        b1, b2 = np.hsplit(bins, [i])  # weights
        # finding means and variances
        m1 = np.sum(p1 * b1) / q1
        m2 = np.sum(p2 * b2) / q2
        v1, v2 = np.sum(((b1 - m1) ** 2) * p1) / q1, np.sum(((b2 - m2) ** 2) * p2) / q2
        # calculates the minimization function
        fn = v1 * q1 + v2 * q2
        if fn < fn_min:
            fn_min = fn
            thresh = i

    _, img_thresh1 = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)
    return img_thresh1

def otsu_binary(img):
     #Otsu binarization function.

    check_2D(img)               #check dim of img - grayscale?        
    blur = cv2.GaussianBlur(img, (KERNEL_SIZE, KERNEL_SIZE), 0)  #gausian blur
    _, otsuImg = cv2.threshold(blur, MIN, MAX, cv2.THRESH_BINARY + cv2.THRESH_OTSU) #find otsu's threshold value with OpenCV function
   
    return otsuImg

def morphology_filter(img, kernelSize):
    #morphology filter

    check_2D(img)    
    morphImgOpen = cv2.morphologyEx(img, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernelSize,kernelSize)))
    morphImgClose = cv2.morphologyEx(morphImgOpen, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernelSize,kernelSize)))
    
    return morphImgClose

def filter_img(img, filterType=0, morphology=False):

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #convert to greyscale
    check_2D(img)                               #check if input image is in grayscale (2D)
    invImg = img_o.invert_image(img)            #some functions are created to work this way
    invImg = invImg[25:210, 0:300]              #make function to crop img, or make function to remove flir bullshit (do the last)

    if filterType == SIMPLE_THRESHOLD_FILTER:   #regular binary threshold
        _, threshImg = cv2.threshold(invImg, THRESHOLD_BINARY, MAX, cv2.THRESH_BINARY) #just regular thresholding with random threshold
    elif filterType == CV_OTZU_FILTER: #openCV otzu threshold
        threshImg = otsu_binary(invImg)
    elif filterType == MANUAL_OTZU_FILTER: #manual calculation of otzu threshold value
        threshImg = manual_otsu_binary(invImg)
    else:
        log.critical('Invalid filter type')
        raise AttributeError('Invalid filter type')
        
    if morphology:
        morphImg = morphology_filter(threshImg, KERNEL_SIZE)
        return morphImg
    return threshImg

#------------------------------   ^FUNCTIONS^   ------------------------------

#------------------------------   v TEST FUNCTION v   ------------------------------
def filters_test_func():
    #just some function to test filters

    print('Lets have a look at all cases of filter func')
    img = img_o.readImage('FLIR0023.jpg')
    cv2.imshow('org', img)

    img1 = filters.filterImg(img, SIMPLE_THRESHOLD_FILTER, False)
    cv2.imshow('thres', img1)

    img2 = filters.filterImg(img, CV_OTZU_FILTER, False)
    cv2.imshow('cv_otz', img2)

    img3 = filters.filterImg(img, MANUAL_OTZU_FILTER, False)
    cv2.imshow('man_otz', img3)

    img4 = filters.filterImg(img, SIMPLE_THRESHOLD_FILTER, True)
    cv2.imshow('thres_morph', img4)

    img5 = filters.filterImg(img, CV_OTZU_FILTER, True)
    cv2.imshow('cv_otz_morph', img5)

    img6 = filters.filterImg(img, MANUAL_OTZU_FILTER, True)
    cv2.imshow('man_otz_morph', img6)

    while True:
        if cv2.waitKey(30) == 27: # exit on ESC
            break
    cv2.destroyAllWindows()

#------------------------------   ^TEST FUNCTION^   ------------------------------

if __name__ == "__main__":
    filters_test_func()
