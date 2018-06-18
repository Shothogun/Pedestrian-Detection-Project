import argparse
import numpy as np
import cv2 as cv



image = cv.imread('test.jpg')

new = cv.cvtColor(image, cv.COLOR_BGR2HSV)

print (image[0][0])
print (new[0][0])

sum_image = cv.sumElems(new)

sum_image_rgb = cv.sumElems(image)

shape = new.shape
#3.344 3.257 
print (sum_image[0] * 2 / (shape[0] * shape[1] * 360))
print (sum_image[1] / (shape[0] * shape[1] * 255))
print ((sum_image_rgb[0] + sum_image_rgb[1] + sum_image_rgb[2]) / (shape[0] * shape[1] * 3 * 255))

cv.imshow('HSV', new)

cv.waitKey(0)
cv.destroyAllWindows()