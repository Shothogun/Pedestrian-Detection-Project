import cv2
import numpy as np
import math

def MultiFrameDif(near_frames):

	size = len(near_frames)

	central_frame = size // 2


	median = []
	diferences = []

	frame_width = near_frames[central_frame].shape[1]
	frame_height = near_frames[central_frame].shape[0]

	if(len(near_frames[central_frame].shape) > 2): # Não é preto e branco

		near_frames[central_frame] = cv2.cvtColor(near_frames[central_frame], cv2.COLOR_RGB2GRAY)

	# Declaring variables
	diference = 7
	diference_result = near_frames [central_frame]
	moving_region = near_frames [central_frame]
	and_op = near_frames [central_frame]

	for position in range(size):

		if (position != central_frame):

			for height in range (frame_height):

				if(len(near_frames[position].shape) > 2): # Não é preto e branco

					near_frames[position] = cv2.cvtColor(near_frames[position], cv2.COLOR_RGB2GRAY)

				for width in range (frame_width):

					threshold = 50

					diference = near_frames[position][height, width] - near_frames[central_frame][height, width]

					if (diference < 0):
						diference = diference * -1

					diference_result[height, width] = diference

					if (diference_result[height, width] < threshold):

						diference_result[height, width] = 0


					else:

						diference_result[height, width] = 1

		diferences.append(diference_result)

	# Median of diference to frames with distance 2 and 1
	cv2.bitwise_and(diferences[0], diferences[len(diferences) - 1], and_op)
	median.append (and_op)

	cv2.bitwise_and(diferences[1], diferences[len(diferences) - 2], and_op)
	median.append (and_op)

	cv2.bitwise_or(median[0], median[1], moving_region)

	return moving_region

#def AdaptiveBackground: