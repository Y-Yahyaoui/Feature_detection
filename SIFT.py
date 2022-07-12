"""
EL OUARDINI CHAIMAE N°22
YAHYAOUI YOUSSEF N°55

SIFT

"""

import numpy as np
import cv2 as cv

# Reading the image
img = cv.imread('test.tif')

# Showing the original image
cv.imshow('img', img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

sift = cv.SIFT_create()

"""
sift.detect() function finds the keypoint in the images.
You can pass a mask if you want to search only a part of image.Each keypoint is a special structure
which has many attributes like its (x,y) coordinates,
size of the meaningful neighbourhood, angle which specifies its orientation, response that specifies strength of keypoints etc
"""
kp = sift.detect(gray, None)

#dst= cv.drawKeypoints(img, kp, img)
dst= cv.drawKeypoints(img, kp, img, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv.imwrite('sift_keypoints.jpg',img)

cv.imshow('dst', img)

if cv.waitKey(0) & 0xff == 27:
    cv.destroyAllWindows()