import cv2
import numpy as np

def readImageFromPath(path, name, ext, amount):
    #Function for reading images from folder. Example: image_5.jpg -> read_image('path', 'image_', 'jpg', 50)
    
    images = []
    for i in range(amount):
        # try:
        img = cv2.imread(path + '/' + name + str(i) + ext, 1)
        # check if image was read
        try:
            if img.shape[0]:
                images.append(img)
        except AttributeError:
            pass
    return images

def readImage(img):
    # read image in greyscale 
    img = cv2.imread(img,cv2.IMREAD_GRAYSCALE)            

    #img = img[40:470, 0:610]
    return img

def invertImage(img):
    #Return inversion of an image.
    return cv2.bitwise_not(img)

if __name__ == "__main__":
    pass