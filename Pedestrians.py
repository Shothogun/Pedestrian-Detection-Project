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
	frame_number = len(Frames_five)/5 

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

	LAO1 = cv.bitwise_and(D[1,:,:],D[2,:,:])
	LAO2 = cv.bitwise_and(D[0,:,:],D[3,:,:])

	MR = np.zeros((height,width), np.uint8)

	MR = cv.bitwise_or(LAO1,LAO2)

	return MR


def main():
	cap = cv.VideoCapture("./Videos/terrace1-c0.avi")
	Frames_five = []
	fgbg = cv.bgsegm.createBackgroundSubtractorMOG()

	while(cap.isOpened()):
		# Capture frame-by-frame
		ret, frame = cap.read()

		if(ret == False):
			break
			
		# Convert color from RGB to Gray
		gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
		height,width = gray.shape

		# Get each frame
		Frames_five.append(gray)

		# For each 5 frames do the operation
		if(len(Frames_five)%5 == 0):

			frame_number = len(Frames_five)/5 

			MR = multi_frame_differecing(Frames_five)

			fgmask = fgbg.apply(frame)

			fgmask = imutils.resize(fgmask, width=min(400, fgmask.shape[1]))

			'''

			# Opening operation in noising videos

			kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(3,3))

			fgmask = cv.morphologyEx(fgmask, cv.MORPH_OPEN, kernel)
			'''

			frame = imutils.resize(frame, width=min(400, frame.shape[1]))

			fgmask = imutils.resize(fgmask, width=min(400, fgmask.shape[1]))

			# Histogram of oriented gradients
			hog = cv.HOGDescriptor()

			hog.setSVMDetector(cv.HOGDescriptor_getDefaultPeopleDetector())

			# Find human shape

			(rects, weights) = hog.detectMultiScale(fgmask, winStride=(4, 4), \
			 padding=(2, 2), scale=1.05)

			rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])

			# Non-max-supression process

			pick = non_max_suppression(rects, probs=None, overlapThresh=0.4)
			for (xA, yA, xB, yB) in pick:
				cv.rectangle(frame, (xA, yA), (xB, yB), (255, 0, 100), 2)

			cv.imshow('frame',frame)	

		# Display the resulting frame
		if cv.waitKey(1) & 0xFF == ord('q'):
			break

	cap.release()
	cv.destroyAllWindows()

if __name__ == "__main__":
	main()