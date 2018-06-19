import cv2
import numpy as np

cap = cv2.VideoCapture("MVI_0099.MOV")

if (cap.isOpened() == False):
	print("Error\n")

while(cap.isOpened()):

	ret, frame = cap.read()

	if (ret==True):

		#frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
		cv2.imshow("Frame", frame)

		if cv2.waitKey(25) & 0xFF == ord('q'):
			break
	else:
		break	

cap.release()

cv2.destroyAllWindows()