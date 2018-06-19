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
	cap = cv.VideoCapture("./Videos/MVI_0126.MOV")
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

			frame_number = int(len(Frames_five)/5)
			MR = emo.multi_frame_differecing(Frames_five)
			fgmask = fgbg.apply(Frames_five[frame_number-3])

			MR += fgmask

			MR = imutils.resize(MR, width=min(400, MR.shape[1]))
		
			'''
			# Test images MR 
			path = "MR_" + str(len(Frames_five)/5) + ".png"
			
			MR = cv.threshold(MR, 40, 255, cv.THRESH_BINARY)[1]

			ret, labels = cv.connectedComponents(MR, 8)

			# Improve the classification of moving objects

			kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(75,75))

			cv.morphologyEx(MR, cv.MORPH_CLOSE, kernel)

			kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(65,65))

			cv.morphologyEx(MR, cv.MORPH_OPEN, kernel)

			#Map components labels to hue value
			if (np.max(labels) != 0):
				label_hue = np.uint8(179*(labels)/np.max(labels))

			else:
				label_hue = np.uint8(179)

			blank_ch = 255*np.ones_like(label_hue)
			labeled_image = cv.merge([label_hue, blank_ch, blank_ch])

			#cvt for display

			labeled_image = cv.cvtColor(labeled_image, cv.COLOR_HSV2BGR)

			#set bg label to black

			labeled_image[label_hue == 0] = 0

			
			cv.imwrite(path, labeled_image)

			if (len(Frames_five) == 800):
				break

		'''
			kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5))

			cv.morphologyEx(MR, cv.MORPH_CLOSE, kernel)

			kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(7,7))

			cv.morphologyEx(MR, cv.MORPH_OPEN, kernel)
	



		fgmask = fgbg.apply(gray)

		kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5))

		fgmask = cv.morphologyEx(fgmask, cv.MORPH_OPEN, kernel)

		frame = imutils.resize(frame, width=min(400, frame.shape[1]))

		fgmask = imutils.resize(fgmask, width=min(400, fgmask.shape[1]))

		hog = cv.HOGDescriptor()

		hog.setSVMDetector(cv.HOGDescriptor_getDefaultPeopleDetector())

		(rects, weights) = hog.detectMultiScale(fgmask, winStride=(4, 4), \
			padding=(8, 8), scale=1.05)


		'''
			for (x, y, w, h) in rects:
				cv.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)

		'''

		rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
		pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

		#pick = sfe.skin_detection(pick, frame)
		#pick = hhe.human_head_ext(pick, frame)

			#print (rects)
			#print (pick)

		
		for (xA, yA, xB, yB) in pick:
			cv.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)
		
		# Display the resulting frame
		cv.imshow('frame',frame)
		if cv.waitKey(1) & 0xFF == ord('q'):
			break

	# When everything done, release the capture
	cap.release()
	cv.destroyAllWindows()

if __name__ == "__main__":
	main()