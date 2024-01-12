import numpy as np
import cv2
import time

training_data = np.load("./final_training_data.npy",allow_pickle=True)
for data in training_data:
	img = data[0]
	choice = data[1]
	cv2.imshow('test', img)
	time.sleep(1)
	print(choice)

	if cv2.waitKey(25) & 0xFF == ord('q'):
		cv2.destroyAllWindows()
		break