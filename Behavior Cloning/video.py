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
        print(next_frame.shape)
        x2, y2 = pyautogui.position()
        cv2.putText(next_frame, str(int(x2)) + ',' + str(int(1080 - y2)), (int(x2), int(y2)), cv2.FONT_HERSHEY_SIMPLEX,
                    2, (0, 0, 0), 4)
        cv2.rectangle(next_frame,(20,1080-600),(800,1080-200),(200,200,200),2)

        frame=next_frame[1080-600:1080-200,20:800]

        cv2.imshow('window', next_frame)
        cv2.imshow('window2', frame)
        print(frame.shape)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break



        count += 1



    video_reader.release()
    return

if __name__ == '__main__':

    drive(PATH_TRAIN_VIDEO, PATH_STEERING_LABELS)
