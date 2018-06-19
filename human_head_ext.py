from imutils.object_detection import non_max_suppression
from imutils import paths
import argparse
import imutils
import numpy as np
import cv2 as cv

def head_detection (rects, frame):

#def head_detection(frames_obj, number_pixels, rect_size):

	position = 0
	pedestrian = []

	for position in range(len(rects)):

		person = frame[rects[position][1]:rects[position][3], rects[position][0]:rects[position][2]]

		new = cv.cvtColor(person, cv.COLOR_BGR2GRAY)

		rect_width = rects[position][2] - rects[position][0]

		rect_height = rects[position][3] - rects[position][1]

		head_size = rect_height * 0.065
		head_size_min = head_size - 0.005
		head_size_max = head_size + 0.005

		head_max_radius = head_size_max / 2
		head_min_radius = head_size_min / 2

		image_circles = cv.HoughCircles (new, cv.HOUGH_GRADIENT, 2, new.scale[0]/4, 200, 100)

		if (number_circles < 4 * head_max_radius and number_circles > 2 * head_min_radius):
			pedestrian.append(rects[position])

	return pedestrian