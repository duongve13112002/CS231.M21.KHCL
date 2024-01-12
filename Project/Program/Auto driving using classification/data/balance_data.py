import numpy as np
import cv2
import pandas as pd
from collections import Counter
from random import shuffle
import os

file_path = "./TRAIN/"
#file_name = 'training_data.npy'

#file = file_path + file_name
starting_value = 1
all_file =[]
while True:
    file_name = 'training_data-{}.npy'.format(starting_value)
    file = file_path + file_name
    if os.path.isfile(file):
        #print('File exists, moving along',starting_value)
        file_load = np.load(file,allow_pickle=True)
        starting_value += 1
        all_file.append(file_load)
    else:
        training_data = np.concatenate(all_file)
        
        break
#training_data = np.load("./training_data.npy",allow_pickle=True)

df = pd.DataFrame(training_data)

print(f"Before: {Counter(df[1].apply(str))}")

lefts = []
rights = []
forwards = []

shuffle(training_data)

for data in training_data:
	img = data[0]
	choice = data[1]

	if choice == [1, 0, 0]:
		lefts.append([img, choice])
	elif choice == [0, 1, 0]:
		forwards.append([img, choice])
	elif choice == [0, 0, 1]:
		rights.append([img, choice])
	else:
		print("no matches")


forwards = forwards[ : len(lefts)][: len(rights)]
lefts = lefts[: len(forwards)]
rights = rights[: len(forwards)]

final_data = forwards + lefts + rights

shuffle(final_data)

print(f"before: {len(training_data)}")
print(f"after: {len(final_data)}")

np.save("./final_training_data.npy", final_data)