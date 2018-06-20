from imutils.object_detection import non_max_suppression
from imutils import paths
import extract_mov_obj as emo
import skin_feature_ext as sfe
import human_head_ext as hhe
import argparse
import imutils
import numpy as np
import cv2 as cv

def main():
	#cap = cv.VideoCapture("./Videos/site0.avi")
	cap = cv.VideoCapture("./Videos/MVI_0114.MOV")
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

			MR = emo.multi_frame_differecing(Frames_five_gray)
			teste_mr = MR.copy()
			fgmask_1 = fgbg.apply(Frames_five[cur_frame])

			MR *= fgmask_1

			ret, MR = cv.threshold(MR, 0, 255, cv.THRESH_BINARY) 

			#MR = imutils.resize(MR, width=min(400, MR.shape[1]))

			ret, labels = cv.connectedComponents(MR, 8)

			# Improve the classification of moving objects

			kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5))

			MR = cv.morphologyEx(MR, cv.MORPH_CLOSE, kernel)

			kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(7,7))

			MR = cv.morphologyEx(MR, cv.MORPH_OPEN, kernel)
			
			labeled_image = Frames_five[cur_frame].copy()

			#Map components labels to hue value
			if (np.max(labels) != 0):
				label_hue = np.uint8(179*(labels)/np.max(labels))

			else:
				label_hue = np.uint8(179)

			blank_ch = 255*np.ones_like(label_hue)

			labeled_image [:,:,0] = label_hue
			labeled_image [:,:,1] = blank_ch
			labeled_image [:,:,2] = blank_ch

			#cvt for display

			labeled_image = cv.cvtColor(labeled_image, cv.COLOR_HSV2BGR)

			#set bg label to black

			labeled_image[label_hue == 0] = 0

			MR = imutils.resize(MR, width=min(550, MR.shape[1]))

			frame_new = imutils.resize(Frames_five[cur_frame], width=min(550, Frames_five[cur_frame].shape[1]))

			MR = imutils.resize(MR, width=min(550, MR.shape[1]))

			# Histogram of oriented gradients
			hog = cv.HOGDescriptor()

			# Find human shape

			hog.setSVMDetector(cv.HOGDescriptor_getDefaultPeopleDetector())


			(rects, weights) = hog.detectMultiScale(MR, winStride=(4, 4), \
			 padding=(2, 2), scale=1.05)

			rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])

			# Non-max-supression process

			pick = non_max_suppression(rects, probs=None, overlapThresh=0.4)

			pick = sfe.skin_detection(pick, frame_new, MR)

			#pick = hhe.head_detection(pick, frame_new, MR)

			for (xA, yA, xB, yB) in pick:
				cv.rectangle(frame_new, (xA, yA), (xB, yB), (0, 255, 0), 2)
			

			Frames_five.remove(Frames_five[0])
			Frames_five_gray.remove(Frames_five_gray[0])


			# Display the resulting frame
			#cv.imshow('frame',fgmask_1)
			#cv.imshow('teste', teste_mr)
			#cv.imshow ('labels', labeled_image)
			cv.imshow('MR', MR)
			cv.imshow('Frame', frame_new)
			
		if cv.waitKey(1) & 0xFF == ord('q'):
			break

	cap.release()
	cv.destroyAllWindows()

if __name__ == "__main__":
	main()