from imutils.object_detection import non_max_suppression
from imutils import paths
import extract_mov_obj as emo
import skin_feature_ext as sfe
import human_head_ext as hhe
import argparse
import imutils
import numpy as np
import cv2 as cv

# It's required to install imutils and pyimagesearch
# for install, type this in the terminal:
# $ pip install pyimagesearch
# $ pip install imutils

# To stop the program press 'q' in the displaying image 

def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("string")
	args = ap.parse_args()

	path = "./Videos/" + args.string

	cap = cv.VideoCapture(path)

	Frames_five = []
	Frames_five_gray = []
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
		Frames_five_gray.append(gray)
		Frames_five.append(frame)

		# For each 5 frames do the operation
		if(len(Frames_five) == 5):
			
			cur_frame = 2

			# Extracting moving objects

			MR = emo.multi_frame_differecing(Frames_five_gray)

			fgmask = fgbg.apply(Frames_five[cur_frame])
			MR *= fgmask

			# Make sure the moving region image is binary

			ret, MR = cv.threshold(MR, 0, 255, cv.THRESH_BINARY) 

			# Improve the classification of moving objects

			kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(7,7))

			MR = cv.morphologyEx(MR, cv.MORPH_CLOSE, kernel)

			kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(7,7))

			MR = cv.morphologyEx(MR, cv.MORPH_OPEN, kernel)
			
			# Resize the image and filter for less computational cost

			MR = imutils.resize(MR, width=min(550, MR.shape[1]))

			frame_new = imutils.resize(Frames_five[cur_frame], width=min(550, Frames_five[cur_frame].shape[1]))

			# Histogram of oriented gradients
			hog = cv.HOGDescriptor()

			# Find human shape

			hog.setSVMDetector(cv.HOGDescriptor_getDefaultPeopleDetector())


			(rects, weights) = hog.detectMultiScale(MR, winStride=(4, 4), \
			 padding=(2, 2), scale=1.05)

			rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])

			# Non-max-supression process

			pick = non_max_suppression(rects, probs=None, overlapThresh=0.4)

			# Improving classification methods

			pick = sfe.skin_detection(pick, frame_new, MR)

			pick = hhe.head_detection(pick, frame_new)

			for (xA, yA, xB, yB) in pick:
				cv.rectangle(frame_new, (xA, yA), (xB, yB), (0, 255, 0), 2)
			

			Frames_five.remove(Frames_five[0])
			Frames_five_gray.remove(Frames_five_gray[0])


			# Display the resulting frame and moving region

			cv.imshow('MR', MR)
			cv.imshow('Frame', frame_new)
			
		if cv.waitKey(1) & 0xFF == ord('q'):
			break

	cap.release()
	cv.destroyAllWindows()

if __name__ == "__main__":
	main()