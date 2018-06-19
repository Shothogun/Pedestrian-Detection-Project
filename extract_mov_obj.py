from imutils.object_detection import non_max_suppression
from imutils import paths
import argparse
import imutils
import numpy as np
import cv2 as cv

def multi_frame_differecing(Frames_five):

	Threshold = 70
	height,width = Frames_five[0].shape

	# Which frame is computed
	frame_number = int(len(Frames_five)/5) 

	# Values especified by the paper
	LAO1 = np.zeros((height,width), np.uint8)
	LAO2 = np.zeros((height,width), np.uint8)
	D = np.zeros((4,height,width), np.uint8)

	D[0] = Frames_five[frame_number-5] - Frames_five[frame_number-3]
	D[1] = Frames_five[frame_number-4] - Frames_five[frame_number-3]
	D[2] = Frames_five[frame_number-2] - Frames_five[frame_number-3]
	D[3] = Frames_five[frame_number-1] - Frames_five[frame_number-3]

	ret, D[0] = cv.threshold(D[0],Threshold,255,cv.THRESH_BINARY)
	ret, D[1] = cv.threshold(D[1],Threshold,255,cv.THRESH_BINARY)
	ret, D[2] = cv.threshold(D[2],Threshold,255,cv.THRESH_BINARY)
	ret, D[3] = cv.threshold(D[3],Threshold,255,cv.THRESH_BINARY)

	LAO1 = D[1,:,:]*D[2,:,:]
	LAO2 = D[0,:,:]*D[3,:,:]

	MR = np.zeros((height,width), np.uint8)

	MR = cv.bitwise_or(LAO1,LAO2)

	return MR