import cv2
import numpy as np
from Grayscale_funcs import*

def readImage():
    # read image 
    #img = cv2.imread('Bilder/Birds.jpg', cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (600,300))
    return img

def blobDetector():
    # Setup SimpleBlobDetector parameters.
    params = cv2.SimpleBlobDetector_Params()

    # Change thresholds
    params.minThreshold = 100
    params.maxThreshold = 255


    # Filter by Area.
    params.filterByArea = True
    params.minArea = 20

    # Filter by Circularity
    params.filterByCircularity = False
    params.minCircularity = 0.1

    # Filter by Convexity
    params.filterByConvexity = False
    params.minConvexity = 0.87

    # Filter by Inertia
    params.filterByInertia = True
    params.minInertiaRatio = 0.01


    #determine cv version
    print(cv2.__version__)
    if cv2.__version__.startswith('2'):
        detector = cv2.SimpleBlobDetector(params)
    else:
        detector = cv2.SimpleBlobDetector_create(params)
    
    return detector

#Diffrent methods of grayscaling image 


def detectStuff(img, detector):
    # detect suff
    keypoints = detector.detect(img)

    #draw detected keypoints as red circles
    imgKeyPoints = cv2.drawKeypoints(img, keypoints, np.array([]),(0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return imgKeyPoints

if __name__ == "__main__":
    img = readImage()
    #img_otsu = adaptive_thresh(img,127,255,"otsu")
    #img_gauss = adaptive_thresh(img,127,255,"gauss")
    #img_mean = adaptive_thresh(img,127,255,"mean")
    img_bin  = adaptive_thresh(img,127,255,"bin")
    detect = blobDetector()

    #newImg_otsu = detectStuff(img_otsu, detect)

    #newImg_gauss = detectStuff(img_gauss, detect)

    #newImg_mean = detectStuff(img_mean, detect)

    newImg_bin = detectStuff(img_bin, detect)

    #display results
    #cv2.imshow("otsu", newImg_otsu)
    #cv2.imshow("gauss",newImg_gauss)
    #cv2.imshow("mean",newImg_mean)
    cv2.imshow("bin",newImg_bin)
    key = cv2.waitKey()
    while key != 27: # exit on ESC
        key = cv2.waitKey()
    
    cv2.destroyAllWindows()