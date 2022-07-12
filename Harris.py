"""
EL OUARDINI CHAIMAE
YAHYAOUI YOUSSEF 

Harris detector

"""
# Importation of libraries, Numpy and OpenCV, the latter is the one that has the harris detector function
import numpy as np
import cv2 as cv

# Reading the image
img = cv.imread('test.tif')

# Showing the original image
cv.imshow('img', img)

"""
Converting the image to a gray scale image to get better results,
because the cv.cornerharris function that we're going 
to use take as an input the grayscale image in a float32 format
that's why we're using np.float32 to convert the gray scale image into floating values
"""
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
gray = np.float32(gray)

"""
Applying the Harris method which takes 4 arguments
1- the image in float32 format
2- the block size(the window), for example we use 2 that means neighbourhoud size is 2, so for each pixel a neighbourhood size of 2*2 pixels is considered
3- the k-size : Aperture parameter of Sobel derivative used. 
The first step is to convert the grayscale image into an image of edges. There are many techniques to do this, but the cv2 uses a filter called Sobel's kernel, which gets cross-correlated with the original image.
The ksize parameter determines the size of the Sobel kernel (3x3, 5x5, etc..). As the size increases, more pixels are part of each convolution process and the edges will get more blurry.
for our case we used 3
4- k- Harris detector free parameter in the equation R=detM-〖k(traceM)〗^2
"""
dst = cv.cornerHarris(gray, 2, 3, 0.04)

# To get the better result we need to dilate the result of the Harris operator 'dst' 
dst = cv.dilate(dst, None)

"""
The dilated image is then reverted to its original form using the optimal threshold value.
Here we consider the product of 0.1 and the maximum value of the dilated image as the optimum threshold.
The points that pass the threshold are marked in red to indicate that they are the detected corners.
"""
img[dst > 0.01 * dst.max()] = [0, 0, 255]

# Showing the result
cv.imshow('dst', img)

if cv.waitKey(0) & 0xff == 27:
    cv.destroyAllWindows()
