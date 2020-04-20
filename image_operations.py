import cv2
import numpy as np
from logging_framework import logging_setup as log

#------------------------------   v FUNCTIONS v   ------------------------------

def read_image_from_path(path, name, ext, amount):
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

def read_image(img):
    # read image in greyscale 
    img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)            

    #img = img[40:470, 0:610]
    return img

def invert_image(img):
    #Return inversion of an image.
    return cv2.bitwise_not(img)

#------------------------------   ^FUNCTIONS^   ------------------------------

if __name__ == "__main__":
    pass