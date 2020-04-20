import cv2
import numpy as np
from . import image_operations as img_o
from logging_framework import logging_setup as log

#global variables
SIMPLE_THRESHOLD_FILTER = 0
CV_OTZU_FILTER = 1
MANUAL_OTZU_FILTER = 2

MORPHOLOGY_ON = True
MORPHOLOGY_OFF = False

def check2D(img):
    # check if input image is in grayscale (2D)
    try:
        if img.shape[2]:
            # if there is 3rd dimension
            print('otsu_binary(img) input image should be in grayscale!')
    except IndexError:
        pass  # image doesn't have 3rd dimension - proceed

def manual_otsu_binary(img):
    # Otsu binarization function by calculating threshold

    check2D(img)

    #gausian blur
    blur = cv2.GaussianBlur(img, (5, 5), 0)

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
    # Otsu binarization function.
    check2D(img)

    #gausian blur
    blur = cv2.GaussianBlur(img, (5, 5), 0)

    # find otsu's threshold value with OpenCV function
    _, otsuImg = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
   
    return otsuImg

def morphologyFilter(img, kernelSize):
    #morphology filter

    check2D(img)

    # check if input image is in grayscale (2D)
    try:
        if img.shape[2]:
            # if there is 3rd dimension
            print('otsu_binary(img) input image should be in grayscale!')
    except IndexError:
        pass  # image doesn't have 3rd dimension - proceed
    

    morphImgOpen = cv2.morphologyEx(img, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernelSize,kernelSize)))
    morphImgClose = cv2.morphologyEx(morphImgOpen, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernelSize,kernelSize)))
    
    return morphImgClose

def filterImg(img, filterType=0, morphology=False):
    # check if input image is in grayscale (2D)
    check2D(img)

    invImg = img_o.invertImage(img) #some functions are created to work this way

    #make function to crop img, or make function to remove flir bullshit (do the last)
    invImg = invImg[20:200, 0:300] #temp

    if filterType == SIMPLE_THRESHOLD_FILTER: #regular binary threshold
        _, threshImg = cv2.threshold(invImg, 60, 255, cv2.THRESH_BINARY) #just regular thresholding with random threshold
    elif filterType == CV_OTZU_FILTER: #openCV otzu threshold
        threshImg = otsu_binary(invImg)
    elif filterType == MANUAL_OTZU_FILTER: #manual calculation of otzu threshold value
        threshImg = manual_otsu_binary(invImg)
        
    else:
        raise AttributeError('Good luck debugging! Gotta love good error messages ;)')
        
    if morphology:
        morphImg = morphologyFilter(threshImg, 5)
        return morphImg
    return threshImg

if __name__ == "__main__":
    pass
