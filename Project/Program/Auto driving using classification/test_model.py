# test_model.py
import sys
sys.path.insert(0, "C:/ĐỒ án các môn HK2 nam 2/Thị Giác máy tính/Đồ án nộp cho thầy/Auto driving using classification/keyloggers")
sys.path.insert(0, "C:/ĐỒ án các môn HK2 nam 2/Thị Giác máy tính/Đồ án nộp cho thầy/Auto driving using classification")
sys.path.insert(0, "C:/ĐỒ án các môn HK2 nam 2/Thị Giác máy tính/Đồ án nộp cho thầy/Auto driving using classification/model")
import numpy as np
from grabscreen import grab_screen
import cv2
import time
from directKeys import PressKey,ReleaseKey, W, A, S, D
from models import models
from getkeys import key_check
from collections import deque, Counter
from capture_data.utility import Utility
import random
from statistics import mode,mean
how_far_remove = 280
rs = (20,15)
log_len = 25 #Lấy 25 - 50

motion_req = 270 # Lấy từ khoảng 260 - 300 là ổn áp nhất ( xác định dc khi nào là bị đụng)
motion_log = deque(maxlen=log_len)

WIDTH = 160
HEIGHT = 120
LR = 1e-3
MODEL_NAME = "gtasa_driving_model"

#0.045 và 0.02 khá ổn
t_time = 0.034
t_time_for_fwd = 0.012
AUTO_STRAIGHT = True
straight_frame_diff = 0

#http://www.noah.org/wiki/movement.py  Nguồn tham khảo tính toán chuyển động
def delta_images(t0, t1, t2):
    d1 = cv2.absdiff(t2, t0)
    return d1
def motion_detection(t_minus, t_now, t_plus,screen):
    delta_view = delta_images(t_minus, t_now, t_plus)
    retval, delta_view = cv2.threshold(delta_view, 16, 255, 3)
    cv2.normalize(delta_view, delta_view, 0, 255, cv2.NORM_MINMAX)
    #img_count_view = cv2.cvtColor(delta_view, cv2.COLOR_RGB2GRAY)
    delta_count = cv2.countNonZero(delta_view)
    dst = cv2.addWeighted(screen,1.0, delta_view,0.6,0)
    delta_count_last = delta_count
    return delta_count
def straight(flag = None):
    # ReleaseKey(A)
    # ReleaseKey(D)
    global AUTO_STRAIGHT
    ReleaseKey(S)
    PressKey(W)
    #time.sleep(random.uniform(0.065,0.09))
    time.sleep(t_time_for_fwd)
    if flag is not None :
        auto_choice = random.randrange(0,3)
        if auto_choice == 0:
            AUTO_STRAIGHT = True
    ReleaseKey(W)

def left():
    global AUTO_STRAIGHT
    # print(straight_frame_diff)
    ReleaseKey(D)
    PressKey(A)
    if AUTO_STRAIGHT:
        PressKey(W)

    #time.sleep(random.uniform(0.065,0.09))
    time.sleep(t_time)    
    ReleaseKey(A)
    ReleaseKey(W)
    ReleaseKey(S)

def right():
    global AUTO_STRAIGHT
    # PressKey(W)
    ReleaseKey(A)
    PressKey(D)
    if AUTO_STRAIGHT:
        PressKey(W)
    
    #time.sleep(random.uniform(0.065,0.09))
    time.sleep(t_time)   
    ReleaseKey(W)
    ReleaseKey(D)
    ReleaseKey(S)
    
def reverse(falls = None):
    if falls is not None:
        falls = random.uniform(0.5,1.2)
    else :
        falls = random.uniform(t_time_for_fwd,t_time_for_fwd*2)   
    #global AUTO_STRAIGHT
    flag = 0 
    #if AUTO_STRAIGHT:
        #AUTO_STRAIGHT = not AUTO_STRAIGHT
        #flag = 1
    PressKey(S)
    time.sleep(falls)
    #if flag == 1:
        #AUTO_STRAIGHT = not AUTO_STRAIGHT
    ReleaseKey(A)
    ReleaseKey(W)
    ReleaseKey(D)


def forward_left(h = None):
    PressKey(W)
    PressKey(A)
    if h is not None:
        time.sleep(0.09)
    else:
        time.sleep(random.uniform(0.5,1.2))    
    ReleaseKey(D)
    ReleaseKey(S)
    
    
def forward_right(h = None):
    PressKey(W)
    PressKey(D)
    if h is not None:
        time.sleep(0.09)
    else:
        time.sleep(random.uniform(0.5,1.2))
    ReleaseKey(A)
    ReleaseKey(S)

    
def reverse_left():
    PressKey(S)
    PressKey(A)
    time.sleep(random.uniform(0.5,1.2))
    ReleaseKey(W)
    ReleaseKey(D)

    
def reverse_right():
    PressKey(S)
    PressKey(D)
    time.sleep(random.uniform(0.5,1.2))
    ReleaseKey(W)
    ReleaseKey(A)

    
model = models(WIDTH, HEIGHT, LR)
model.load(f"./model/saved_model/{MODEL_NAME}")
stop_run = 1
def main():
    

    paused = False
    recurs = True
    straight_count = 0
    mode_choice = 0
    frame_number = 0
    # Getting roi coordinates
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
    #print(x0,y0,x1,y1)
    global AUTO_STRAIGHT
    last_time = time.time()
    for i in list(range(10))[::-1]:
        print(i+1)
        time.sleep(1)
    screen = grab_screen(region=(x0,y0,x1,y1))
    screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
    prev = cv2.resize(screen, (160,120))
    t_minus = prev
    t_now = prev
    t_plus = prev
    dem_a_d = 0 
    a_key = 0
    d_key = 0 
    while(True):
        
        frame_number += 1

        if not paused:
            screen = grab_screen(region=(x0,y0,x1,y1))
            # print('loop took {} seconds'.format(time.time()-last_time))
            last_time = time.time()
            screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
            screen = cv2.resize(screen, (160,120))

            delta_count_last = motion_detection(t_minus, t_now, t_plus,screen)
            t_minus = t_now
            t_now = t_plus
            t_plus = screen
            t_plus = cv2.blur(t_plus,(4,4))
            prediction = model.predict([screen.reshape(160,120,1)])[0]
            print(prediction)

            turn_right_thresh = 0.65
            turn_left_thresh = 0.65
            straight_thresh = 0.6

            pred_max = np.argmax(prediction)
            #print(pred_max)

            if frame_number % 60 == 0:
                AUTO_STRAIGHT = not AUTO_STRAIGHT
                print("disable AUTO_STRAIGHT")
                frame_number = 0
            print('AUTO_STRAIGHT is ',AUTO_STRAIGHT)
            #print('dem_a_d :',dem_a_d)
            if dem_a_d > 16:
                 dem_a_d = 0
                 AUTO_STRAIGHT = True               
            if pred_max == 0 and prediction[0]>= turn_left_thresh:
                # print(straight_frame_diff)
                left()
                dem_a_d +=1
                straight_count = 0
                a_key+=1
                print("a")
            elif pred_max == 1 and prediction[1]>= straight_thresh:
                straight()
                dem_a_d = 0
                straight_count+=1
                a_key = 0
                d_key = 0                
                print("w")
            elif pred_max == 2 and prediction[2]>= turn_right_thresh:
                right()
                dem_a_d +=1
                straight_count = 0 
                d_key+=1
                print("d")
            else:
                ReleaseKey(W)
                ReleaseKey(A)
                ReleaseKey(D)
                ReleaseKey(S)
                time.sleep(0.05)    
            motion_log.append(delta_count_last)
            motion_avg = round(mean(motion_log),3)
            print('motion_avg: ',motion_avg)
            if motion_avg < motion_req  and len(motion_log) >= log_len and recurs is True :
                print('WERE PROBABLY STUCK FFS.')
                quick_choice = random.randrange(2,6)
                
                #if quick_choice == 0:
                if a_key < d_key:
                    reverse(1)
                    #time.sleep(random.uniform(0,0.5))
                    forward_left()
                    #time.sleep(random.uniform(0,0.5))
                    print('s and wa')
                    a_key = d_key = 0 
                #elif quick_choice == 1:
                elif a_key > d_key:
                    reverse(1)
                    #time.sleep(random.uniform(0,0.5))
                    forward_right()
                    #time.sleep(random.uniform(0,0.5))
                    print('s and wd')
                    a_key = d_key = 0 
                elif a_key == d_key: 
                    reverse(1)
                    a_key = d_key = 0 
                    print('s ')       
                elif quick_choice == 2:
                    reverse_left()
                    #time.sleep(random.uniform(0,0.5))
                    forward_right()
                    #time.sleep(random.uniform(0,0.5))

                elif quick_choice == 3:
                    reverse_right()
                    #time.sleep(random.uniform(0,0.5))
                    forward_left()
                    #time.sleep(random.uniform(0,0.5))
                else:
                    reverse(1)
                    #time.sleep(random.uniform(0,0.5))

                for i in range(log_len-2):
                    del motion_log[0]
        keys = key_check()

        # p pauses game and can get annoying.
        if recurs is False and motion_avg < motion_req and len(motion_log) >= log_len:
            for i in range(log_len-2):
                del motion_log[0]
        if 'P' in keys:
            recurs = not recurs            
        if 'T' in keys:
            if paused:
                paused = False
                time.sleep(1)
            else:
                paused = True
                ReleaseKey(A)
                ReleaseKey(W)
                ReleaseKey(D)
                ReleaseKey(S)
                time.sleep(1)
        if 'E' in keys:
            break

main()       
