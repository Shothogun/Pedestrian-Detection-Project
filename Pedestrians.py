from imutils.object_detection import non_max_suppression
from imutils import paths
import argparse
import imutils
import numpy as np
import cv2 as cv


# It's required to install imutils and pyimagesearch
# for install, type this in the terminal:
# $ pyp install pyimagesearch
# $ pyp install imutils


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("string")
	args = ap.parse_args()

	path = "./Videos/" + args.string

	cap = cv.VideoCapture(path)
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
		# (Video's frames sincronization)
		if(len(Frames_five)%3 == 0):

			fgmask = fgbg.apply(frame)
			
			'''
			# Opening operation in noising videos

			kernel_A = cv.getStructuringElement(cv.MORPH_ELLIPSE,(3,3))

			kernel_B = cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5))			
			
			fgmask = cv.morphologyEx(fgmask, cv.MORPH_OPEN, kernel_A)	

			fgmask = cv.morphologyEx(fgmask, cv.MORPH_CLOSE, kernel_B)	

			'''

			frame = imutils.resize(frame, width=min(550, frame.shape[1]))

			fgmask = imutils.resize(fgmask, width=min(550, fgmask.shape[1]))

			# Histogram of oriented gradients
			hog = cv.HOGDescriptor()

			# Find human shape

			hog.setSVMDetector(cv.HOGDescriptor_getDefaultPeopleDetector())

			(rects, weights) = hog.detectMultiScale(fgmask, winStride=(4, 4), \
			 padding=(2, 2), scale=1.05)

			rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
			
			# Non-max-supression process

			pick = non_max_suppression(rects, probs=None, overlapThresh=0.4)

			
			for (xA, yA, xB, yB) in pick:
				cv.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)

			cv.imshow('frame',frame)	

		# Display the resulting frame
		if cv.waitKey(1) & 0xFF == ord('q'):
			break

	cap.release()
	cv.destroyAllWindows()

if __name__ == "__main__":
	main()