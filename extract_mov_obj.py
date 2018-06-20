from imutils.object_detection import non_max_suppression
from imutils import paths
import argparse
import imutils
import numpy as np
import cv2 as cv

def multi_frame_differecing(Frames_five):

	Threshold = 180
	height,width = Frames_five[0].shape

	# Which frame is computed
	cur_frame = 2 

	# Values especified by the paper
	LAO1 = np.zeros((height,width), np.uint8)
	LAO2 = np.zeros((height,width), np.uint8)
	D = np.zeros((4,height,width), np.float32)
	Dif = np.zeros((4,height, width), np.float32)

	D[0] = Frames_five[cur_frame-2] - Frames_five[cur_frame]
	D[1] = Frames_five[cur_frame-1] - Frames_five[cur_frame]
	D[2] = Frames_five[cur_frame+1] - Frames_five[cur_frame]
	D[3] = Frames_five[cur_frame+2] - Frames_five[cur_frame]

	Dif[0] = cv.sqrt(cv.pow(D[0], 2))
	Dif[1] = cv.sqrt(cv.pow(D[1], 2))
	Dif[2] = cv.sqrt(cv.pow(D[2], 2))
	Dif[3] = cv.sqrt(cv.pow(D[3], 2))

	Dif[0] = (Dif[0]).astype('uint8')
	Dif[1] = (Dif[1]).astype('uint8')
	Dif[2] = (Dif[2]).astype('uint8')
	Dif[3] = (Dif[3]).astype('uint8')

	ret, Dif[0] = cv.threshold(Dif[0],Threshold,255,cv.THRESH_BINARY)
	ret, Dif[1] = cv.threshold(Dif[1],Threshold,255,cv.THRESH_BINARY)
	ret, Dif[2] = cv.threshold(Dif[2],Threshold,255,cv.THRESH_BINARY)
	ret, Dif[3] = cv.threshold(Dif[3],Threshold,255,cv.THRESH_BINARY)

	#cv.imshow('D0', Dif[0])
	#cv.imshow('D1', Dif[1])

	LAO1 = D[1,:,:]*D[2,:,:]
	LAO2 = D[0,:,:]*D[3,:,:]

	MR = np.zeros((height,width), np.uint8)

	MR = cv.bitwise_or(LAO1,LAO2)

	MR = MR.astype('uint8')

	return MR