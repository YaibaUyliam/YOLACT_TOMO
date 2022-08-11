import cv2
import os
import numpy as np


def gausian_blur(image):
    image = cv2.GaussianBlur(image,(3,3),cv2.BORDER_TRANSPARENT)
    return image

def averageing_blur(image):
    image=cv2.blur(image,(3,3))
    return image

def dilation_image(image,shift=3):
    kernel = np.ones((shift, shift), np.uint8)
    image = cv2.dilate(image,kernel,iterations = 1)
    return image

def erosion_image(image,shift=3):
    kernel = np.ones((shift,shift),np.uint8)
    image = cv2.erode(image,kernel,iterations = 1)
    return image

def sharpen_image(image):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    image = cv2.filter2D(image, -1, kernel)
    return image

def augment_data(image):

    Augment_list=[gausian_blur,averageing_blur,dilation_image,erosion_image,sharpen_image]

    # Random to choose augmentation way
    image = np.random.choice(Augment_list)(image)
    return image

