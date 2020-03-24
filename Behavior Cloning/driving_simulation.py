import cv2
import os, time, sys, shutil
import numpy as np
import math
import pyautogui

PATH_DATA_FOLDER = './data/'


PATH_TRAIN_VIDEO = '/home2/BRITE/mapping-and-localization2/src/BehaviorCloning/train_steering_angle_dataset.mp4'
PATH_STEERING_LABELS = PATH_DATA_FOLDER +  'steering_angles.txt'


def get_angle(x1,y1,x2,y2):
    angle=math.atan(((y2-y1)/(x2-x1)))

    return angle*180/3.14




def drive(video_input_path, steering_angle_output):

    video_reader = cv2.VideoCapture(video_input_path)

    count = 0
    pyautogui.moveTo(301, 745)
    file = open("steering_ground_truth.txt", "a")
    while True:
        ret, next_frame = video_reader.read()
        next_frame = cv2.resize(next_frame, (1920, 1080))
        cv2.circle(next_frame,(300,845),100,(200,200,200),2)

        x1=300
        y1=845
        x2,y2=pyautogui.position()

        if (x1==x2):
            angle=90
        else:
            angle= get_angle(x1,1080-y1,x2,1080-y2)
        angle=abs(angle)
        if(x2>x1):
            angle=90-abs(angle)
        else:
            angle=abs(angle)-90
        #cv2.putText(next_frame,str(int(x2))+','+str(int(1080-y2)),(int(x1),int(y1)),cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 4)
        cv2.putText(next_frame,str(angle),(int(x2),int(y2)),cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 4)

        # plot steering wheel line
        pt1x1=x1
        pt1y1=1080-y1
        pt2y2=100*math.sin(math.radians(-angle+90))+pt1y1
        pt2x2=100*math.cos(math.radians(-angle+90))+pt1x1

        cv2.line(next_frame,(300,845),(int(pt2x2),int(1080-pt2y2)),(200,200,200),4)
        file.writelines(str(angle)+'\n')

        if ret is False:
            break


        cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("window", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow('window', next_frame)


        print(angle)
        print(x2,y2)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break



        count += 1



    video_reader.release()
    return

if __name__ == '__main__':

    drive(PATH_TRAIN_VIDEO, PATH_STEERING_LABELS)
