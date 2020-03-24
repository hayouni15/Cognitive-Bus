from model import CNNModel
import cv2
import sys
import time
import numpy as np
import math

import matplotlib.pyplot as plt

PATH_DATA_FOLDER = './data/'
PATH_TEST_LABEL = PATH_DATA_FOLDER +  'result.txt'
PATH_TEST_VIDEO = '/home2/BRITE/mapping-and-localization2/src/BehaviorCloning/train_steering_angle_dataset.mp4'
PATH_TEST_VIDEO='/home/ahayouni/Documents/brite-unit2/src/speedEstimation/data/brite_train.mp4'

PATH_TEST_VIDEO_OUTPUT = PATH_DATA_FOLDER + 'test_portion_augnmented_output.mp4'
PATH_COMBINED_TEST_VIDEO_OUTPUT = PATH_DATA_FOLDER + 'combined_test_output.mp4'


PATH_TRUE_steering = 'steering_ground_truth.txt'


TYPE_FLOW_PRECOMPUTED = 0
TYPE_ORIGINAL = 1


MODEL_NAME = 'brite_biased_mph_CNNModel_flow'
MODEL_NAME='brite_steering'
#MODEL_NAME = 'bestCNNModel_flow2'

PRE_TRAINED_WEIGHTS = './'+MODEL_NAME+'.h5'

def img_preprocess(img):

    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img,  (3, 3), 0)
    img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)

    return img

def draw_curved_points(frame,pts_number,angle,x1,y1,color):
    y1=1080-845
    if angle<0:
        angle=90+abs(angle)
    else:
        angle = 90-angle
    for i in range (pts_number):
        y2 = y1 + i
        coef=0.064-0.007*i+0.000159*i*i
        x2=x1+((y2-y1)/math.tan(math.radians(angle)))*coef
        cv2.circle(frame, (int(x2), 1080-int(y2)), 2, color, 2)

def predict_from_video(video_input_path,Ground_truth, original_video_output_path, combined_video_output_path):
    predicted_labels = []

    video_reader = cv2.VideoCapture(video_input_path)

    num_frames = video_reader.get(cv2.CAP_PROP_FRAME_COUNT)
    frame_size = (int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps = int(video_reader.get(cv2.CAP_PROP_FPS))

    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fourcc = 0x00000021
    video_writer = cv2.VideoWriter(original_video_output_path, fourcc, fps, frame_size)
    video_writer_combined = cv2.VideoWriter(combined_video_output_path, fourcc, fps, frame_size)





    predicted_labels.append(0.0)

    file1 = open(PATH_TEST_LABEL, "a+")



    font                   = cv2.FONT_HERSHEY_SIMPLEX
    place = (300, 50)
    place2 = (300, 75)

    fontScale              = 1
    fontColor              = (255,255,255)
    lineType               = 2

    count =0
    while True:
        ret, next_frame = video_reader.read()

        if ret is False:
            break
        t_start = time.time()

        next_frame = cv2.resize(next_frame, (1920, 1080))
        driving_window = next_frame[1080 - 600:1080 - 200, 20:800]
        driving_window = img_preprocess(driving_window)



        combined_image_test = cv2.normalize(driving_window, None, alpha=-1, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)



        combined_image_test = combined_image_test.reshape(1, combined_image_test.shape[0], combined_image_test.shape[1], combined_image_test.shape[2])


        prediction = model.predict(combined_image_test)

        pt1x1=x1
        pt1y1=1080-y1
        pt2y2 = 100 * math.sin(math.radians(-(prediction)[0][0] + 90)) + pt1y1
        pt2x2 = 100 * math.cos(math.radians(-(prediction) + 90)) + pt1x1


        truth = float(Ground_truth[count])
        #cv2.line(next_frame, (300, 845), (int(pt2x2), int(1080 - pt2y2)), (200, 200, 200), 4)
        pty2 = 100 * math.sin(math.radians(-truth + 90)) + pt1y1
        ptx2 = 100 * math.cos(math.radians(-truth + 90)) + pt1x1

        #cv2.line(next_frame, (300, 845), (int(ptx2), int(1080 - pty2)), (200, 200, 200), 4)
        # if count>5:
        #     prediction[0][0]=(prediction[0][0] + predicted_labels[count - 1] + predicted_labels[count - 2] + predicted_labels[count - 3]) / 4


        predicted_labels.append(prediction[0][0])
        #draw_curved_points(next_frame, 100, truth, 300, 845, (0, 255, 0))
        draw_curved_points(next_frame,100,prediction[0][0],300,845,(255,0,0))


        file1.writelines(str(prediction[0][0])+ '\n')






        cv2.putText(next_frame, str(int(prediction[0][0])), place, font, fontScale,(0,0,255),lineType)
        cv2.putText(next_frame, str(int(truth)), place2, font, fontScale, (0,255,0), lineType)


        cv2.imshow('frame',next_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        video_writer.write(next_frame)



        count +=1
        sys.stdout.write('\rprocessed frames: %d of %d' % (count, num_frames))
        t_end=time.time()





    t2 = time.time()
    video_reader.release()
    video_writer.release()
    video_writer_combined.release()
    file1.close()
    print(' Prediction completed !')



    return predicted_labels

def load_GT(PATH_TRUE_SPEED):
    f = open(PATH_TRUE_SPEED, "r")
    GT=[]
    for x in f:
        date_string = x.split('\n')[0]
        GT.append(date_string)
    return GT


if __name__ == '__main__':
    x1 = 300
    y1 = 845

    model = CNNModel()
    model.load_weights(PRE_TRAINED_WEIGHTS)

    print('Testing model...')
    Ground_truth=load_GT(PATH_TRUE_steering)
    predicted_labels = predict_from_video(PATH_TEST_VIDEO,Ground_truth,  PATH_TEST_VIDEO_OUTPUT, PATH_COMBINED_TEST_VIDEO_OUTPUT)

    # with open(PATH_TEST_LABEL, mode="w") as outfile:
    #     for label in predicted_labels:
    #         outfile.write("%s\n" % str(label))
