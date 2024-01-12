# create_training_data.py
import sys
sys.path.insert(0, "C:/ĐỒ án các môn HK2 nam 2/Thị Giác máy tính/Đồ án nộp cho thầy/Auto driving using classification/keyloggers")
sys.path.insert(0, "C:/ĐỒ án các môn HK2 nam 2/Thị Giác máy tính/Đồ án nộp cho thầy/Auto driving using classification")
sys.path.insert(0, "C:/ĐỒ án các môn HK2 nam 2/Thị Giác máy tính/Đồ án nộp cho thầy/Auto driving using classification/model")
import numpy as np
from grabscreen import grab_screen
import cv2
import time
from getkeys import key_check
import os
from capture_data.utility import Utility

def keys_to_output(keys):
    '''
    Convert keys to a ...multi-hot... array
    [A,W,D] boolean values.
    '''
    output = [0,0,0]
    
    if 'A' in keys:
        output[0] = 1
    elif 'D' in keys:
        output[2] = 1
    elif 'W' in keys:
        output[1] = 1
    return output


file_path = "./data/TRAIN/"
#file_name = 'training_data.npy'

#file = file_path + file_name
starting_value = 1

while True:
    file_name = 'training_data-{}.npy'.format(starting_value)
    file = file_path + file_name
    if os.path.isfile(file):
        print('File exists, moving along',starting_value)
        starting_value += 1
    else:
        print('File does not exist, starting fresh!',starting_value)
        
        break
'''if os.path.isfile(file):
    print('File exists, loading previous data!')
    training_data = list(np.load(file,allow_pickle=True))
else:
    print('File does not exist, starting fresh!')
    training_data = []'''


def main(file, starting_value):
    file = file
    starting_value = starting_value
    training_data = []
    print(
        "\nDrag the mouse to the region which needs to be recorded and press enter...\nEnter 'c' to cancel!"
    )
    util = Utility()
    try:
        x0, y0, x1, y1 = util.get_coordinates()  # returns 4 coordinates
        if x0 == 0 and y0 == 0 and x1 == 0 and y0 == 0:
            sys.exit("ROI not selected")
    except BaseException as error:
        print("\nLog : ", error)
        print(
            "The Region of Intrest was not selected properly. Please rerun the program\n"
        )
        sys.exit()
    for i in list(range(6))[::-1]:
        print(i+1)
        time.sleep(1)
    print("data collection started..")

    paused = False
    while(True):

        if not paused:
            game_screen = grab_screen(region=(x0,y0,x1,y1))
            last_time = time.time()
            game_screen = cv2.cvtColor(game_screen, cv2.COLOR_BGR2GRAY)
            game_screen = cv2.resize(game_screen, (160,120))
            # resize to something a bit more acceptable for a CNN
            keys = key_check()
            output = keys_to_output(keys)

            training_data.append([game_screen, output])
            
            if len(training_data) % 500 == 0:
                print(len(training_data)," " ,starting_value)
                np.save(file,training_data)
                training_data = []
                print('SAVED')
                starting_value += 1
                file = './data/TRAIN/training_data-{}.npy'.format(starting_value)

        keys = key_check()
        # print(keys)
        if 'T' in keys:
            if paused:
                paused = False
                print('unpaused!')
                time.sleep(1)
            else:
                print('Pausing!')
                paused = True
                time.sleep(1)
        if 'E' in keys:
            break


main(file,starting_value)
print("data collected.")
