import cv2


def adaptive_thresh(img,thresh,max,type):

    if(type == "bin"):
        #Normal binary thresholding 

        ret,th1 = cv2.threshold(img,thresh,max,cv2.THRESH_BINARY) 
        return th1
    elif(type == "mean"):
        #Adaptive mean thresholding

        img_blur = cv2.medianBlur(img,5)
        ret,th1 = cv2.threshold(img_blur,thresh,max,cv2.THRESH_BINARY)
        th2 = cv2.adaptiveThreshold(th1,max,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,21,0)
        return th2
    elif(type =="gauss"):
        #Adaptive gaussian thresholding 

        img_blur = cv2.medianBlur(img,5)
        ret,th1 = cv2.threshold(img_blur,thresh,max,cv2.THRESH_BINARY)
        th3 = cv2.adaptiveThreshold(th1,max,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,5,0)
        return th3
    else:
        #Otsu thersholding 
        blur = cv2.GaussianBlur(img,(5,5),0)
        ret,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        return th3

