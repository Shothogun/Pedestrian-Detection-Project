import cv2
import numpy as np
import extract_mov_obj as emo

def make_360p():
	cap.set(3, 640)
	cap.set(4, 360)

def rescale_frame(frame, percent=75):
	scale_percent = percent
	width = int(frame.shape[1] * scale_percent / 100)
	height = int(frame.shape[0] * scale_percent / 100)
	dim = (width, height)
	return cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)

cap = cv2.VideoCapture("Videos/corte.wmv")

make_360p() # Diminui a resolucao (teoricamente)

if (cap.isOpened() == False):
	print("Error\n")

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

out = cv2.VideoWriter('test.avi',cv2.VideoWriter_fourcc(*'XVID'), 25, (frame_width,frame_height))

ret, prev_frame = cap.read()

prev_frame = rescale_frame(prev_frame) # Diminui o tamanho do frame

out.write(prev_frame)

near_frames = [prev_frame]

while(cap.isOpened()):

	ret, cur_frame = cap.read()

	cur_frame = rescale_frame(cur_frame)

	if (ret==True):

		near_frames.append(cur_frame)

		if (len(near_frames) < 3 ): # Printa ate o terceiro frame

			cv2.imshow("Frame", cur_frame)
			out.write(cur_frame)

		elif (len(near_frames) >= 5 ): # Processa as imagens em relacao ao frame central e prepara para receber o proximo frame

			center_frame = len(near_frames) // 2

			new_frame = emo.MultiFrameDif(near_frames)

			near_frames.remove(near_frames[0])

			new_frame = near_frames[center_frame] * new_frame

			out.write(new_frame)
			cv2.imshow("Frame", new_frame)

		#frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

		if cv2.waitKey(25) & 0xFF == ord('q'):
			break
	else:
		break	

	prev_frame = cur_frame

cap.release()
out.release()

cv2.destroyAllWindows()