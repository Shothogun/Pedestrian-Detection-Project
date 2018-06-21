from imutils.object_detection import non_max_suppression
from imutils import paths
import argparse
import imutils
import numpy as np
import math as mt
import cv2 as cv

def head_detection (rects, frame):

	position = 0
	pedestrian = []

	#Process each moving region of the image
	for position in range(len(rects)):

		person = frame[rects[position][1]:rects[position][3], rects[position][0]:rects[position][2]]

		new = cv.cvtColor(person, cv.COLOR_BGR2GRAY)

		rect_width = rects[position][2] - rects[position][0]

		rect_height = rects[position][3] - rects[position][1]

		head_size = rect_height * 0.065
		head_size_min = head_size - 0.005
		head_size_max = head_size + 0.005

		head_max_radius = mt.floor(head_size_max / 2)
		head_min_radius = mt.floor(head_size_min / 2)

		image_circles = cv.HoughCircles (new, cv.HOUGH_GRADIENT, 2, new.shape[0]/4, 200,100, int(head_min_radius), int(head_max_radius))

		number_circles = image_circles.shape[1]

		if (number_circles <= 4 * head_max_radius and number_circles >= 2 * head_min_radius - 1):
			pedestrian.append(rects[position])

	return pedestrian