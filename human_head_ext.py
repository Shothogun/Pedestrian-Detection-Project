from imutils.object_detection import non_max_suppression
from imutils import paths
import argparse
import imutils
import numpy as np
import cv2 as cv

def head_detection(frames_obj, number_pixels, rect_size):

	position = 0
	pedestrian = []

	for position in range(len(frames_obj)):

		new = cv.cvtColor(frames_obj[position], cv.COLOR_BGR2GRAY)

		head_size = rect_size.height[position] * 0.065
		head_size_min = head_size - 0.005
		head_size_max = head_size + 0.005

		head_max_radius = head_size_max / 2
		head_min_radius = head_size_min / 2

		image_circles = cv.HoughCircles (new,        )

		if (number_circles < 4 * head_max_radius and number_circles > 2 * head_min_radius):
			pedestrian.append(frames_obj[position])