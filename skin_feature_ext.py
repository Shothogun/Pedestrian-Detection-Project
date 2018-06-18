from imutils.object_detection import non_max_suppression
from imutils import paths
import argparse
import imutils
import numpy as np
import cv2 as cv

def skin_detection(frames_obj, number_pixels, rect_prop):

	constant = 3.3
	min_H = (constant - 0.15)/10 - 0.012
	max_H = (constant - 0.15)/10 + 0.012
	min_S = (constant + 0.1)/10 - 0.7
	max_S = (constant + 0.1)/10 + 0.7
	position = 0
	pedestrians = []

	for position in range(len(frames_obj)):

		new = cv.cvtColor(frames_obj[position], cv.COLOR_BGR2HSV)

		sum_image = cv.sumElems(new)

		sum_H = sum_image[0]
		sum_S = sum_image[1]
		#sum_I = sum_image[2]

		med_H = sum_H / number_pixels[position]
		med_S = sum_S / number_pixels[position]
		#med_I = sum_I / number_pixels[position]

			if (med_H > min_H and med_H < max_H and med_S > min_S and med_S < max_S): #Skin color characteristics

				if (rect_prop < 0.357 and rect_prop > 0.231):
					pedestrians.append(frames_obj[position])

		return pedestrians


