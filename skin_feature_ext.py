from imutils.object_detection import non_max_suppression
from imutils import paths
import argparse
import imutils
import numpy as np
import cv2 as cv
import math as mt

#def skin_detection(frames_obj, number_pixels, rect_prop):

def skin_detection (rects, frame, MR):

	#constant = 3.3

	min_H = 0
	max_H = 50
	min_S = 0.23
	max_S = 0.68
	min_R = 95
	min_G = 40
	min_B = 20

	position = 0
	pedestrians = []

	MR_3 = frame.copy()

	MR_3 [:,:,0] = MR
	MR_3 [:,:,1] = MR
	MR_3 [:,:,2] = MR

	region = cv.multiply(frame, MR_3)

	for position in range(len(rects)):

		person = region[rects[position][1]:rects[position][3], rects[position][0]:rects[position][2]]

		#if (person.shape[1] != 0 and person.shape[0] != 0):
			#cv.imshow('LuL',person)
			#cv.waitKey(1)

		rect_width = rects[position][2] - rects[position][0]

		rect_height = rects[position][3] - rects[position][1]

		#number_pixels = rect_height * rect_width

		rect_prop = rect_width/rect_height

		new = cv.cvtColor(person, cv.COLOR_BGR2HSV)

		found = 0

		for height in range(rects[position][1], rects[position][3]):

			for width in range (rects[position][0], rects[position][2]):

				color_HSV = new[height - rects[position][1], width - rects[position][0]]

				if (color_HSV[0]*2 > min_H and color_HSV[0]*2 < max_H and color_HSV[1]/255 > min_S and color_HSV[1]/255 < max_S):

					color_BGR = person[height - rects[position][1], width - rects[position][0]]

					if( color_BGR[2] > min_R and color_BGR[0] > min_B and color_BGR[1] > min_G and max(color_BGR) == color_BGR[2] 
						and mt.sqrt(mt.pow(color_BGR[2],2) - mt.pow(color_BGR[1],2)) > 15 ):

						#if (rect_prop < 0.357 and rect_prop > 0.231):

						pedestrians.append(rects[position])
						found = found + 1
						if (found == 50):
							break

			if (found == 50):
				break
		'''
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
'''
	return pedestrians