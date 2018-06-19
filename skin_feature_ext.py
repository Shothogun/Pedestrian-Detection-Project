from imutils.object_detection import non_max_suppression
from imutils import paths
import argparse
import imutils
import numpy as np
import cv2 as cv

#def skin_detection(frames_obj, number_pixels, rect_prop):

def skin_detection (rects, frame):

	constant = 3.3
	min_H = (constant - 0.15)/10 - 0.012
	max_H = (constant - 0.15)/10 + 0.012
	min_S = (constant + 0.1)/10 - 0.7
	max_S = (constant + 0.1)/10 + 0.7
	position = 0
	pedestrians = []

	for position in range(len(rects)):

		person = frame[rects[position][1]:rects[position][3], rects[position][0]:rects[position][2]]

		if (person.shape[1] != 0 and person.shape[0] != 0):
			cv.imshow('LuL',person)
			cv.waitKey(1)

		rect_width = rects[position][2] - rects[position][0]

		rect_height = rects[position][3] - rects[position][1]

		number_pixels = rect_height * rect_width

		rect_prop = rect_width/rect_height

		new = cv.cvtColor(person, cv.COLOR_BGR2HSV)

		sum_image = cv.sumElems(new)

		sum_H = sum_image[0]
		sum_S = sum_image[1]
		#sum_I = sum_image[2]

		med_H = sum_H * 2 / (number_pixels * 360)
		med_S = sum_S / (number_pixels * 255)
		#med_I = sum_I / number_pixels[position]

		print (med_H)
		print (med_S)
		print (rect_prop)
		#if (med_H > min_H and med_H < max_H and med_S > min_S and med_S < max_S): #Skin color characteristics

		if (rect_prop < 0.357 and rect_prop > 0.231):
			pedestrians.append(rects[position])

	return pedestrians


