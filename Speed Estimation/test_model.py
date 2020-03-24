from model import CNNModel
import cv2
import sys
import time
import numpy as np
from frames_to_opticalFlow import convertToOptical
import matplotlib.pyplot as plt

PATH_DATA_FOLDER = './data/'
PATH_TEST_LABEL = PATH_DATA_FOLDER +  'brite_mph_train_output.txt'
PATH_TEST_VIDEO = PATH_DATA_FOLDER + 'brite_train.mp4'
PATH_TEST_VIDEO_OUTPUT = PATH_DATA_FOLDER + 'test_portion_augnmented_output.mp4'
PATH_COMBINED_TEST_VIDEO_OUTPUT = PATH_DATA_FOLDER + 'combined_test_output.mp4'


PATH_TRUE_SPEED = PATH_DATA_FOLDER +  'brite_mph_train.txt'


TYPE_FLOW_PRECOMPUTED = 0
TYPE_ORIGINAL = 1


MODEL_NAME = 'brite_biased_mph_CNNModel_flow'
#MODEL_NAME = 'bestCNNModel_flow2'

PRE_TRAINED_WEIGHTS = './'+MODEL_NAME+'.h5'


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

    t1 = time.time()
    ret, prev_frame = video_reader.read()
    prev_frame = cv2.resize(prev_frame, (640, 480))
   # prev_frame = cv2.resize(prev_frame, (640, 480))
    hsv = np.zeros_like(prev_frame)

    video_writer.write(prev_frame)

    predicted_labels.append(0.0)

    flow_image_bgr_prev1 =  np.zeros_like(prev_frame)
    #flow_image_bgr_prev2 =  np.zeros_like(prev_frame)
   # flow_image_bgr_prev3 =  np.zeros_like(prev_frame)
    #flow_image_bgr_prev4 =  np.zeros_like(prev_frame)

    prediction_output_file=file1 = open(PATH_TEST_LABEL, "a+")



    font                   = cv2.FONT_HERSHEY_SIMPLEX
    place = (300, 50)
    place2 = (300, 75)
    place3 = (50, 50)
    place4 = (50, 75)
    fontScale              = 1
    fontColor              = (255,255,255)
    lineType               = 2

    count =0
    while True:
        ret, next_frame = video_reader.read()

        if ret is False:
            break
        t_start = time.time()

        next_frame = cv2.resize(next_frame, (640, 480))
        #next_frame = cv2.resize(next_frame, (640, 480))
        flow_image_bgr_next = convertToOptical(prev_frame, next_frame)
        t_after_optixcal_conversion = time.time()
        #flow_image_bgr = (flow_image_bgr_prev1 + flow_image_bgr_prev2 +flow_image_bgr_prev3 +flow_image_bgr_prev4 + flow_image_bgr_next)/4
        flow_image_bgr=flow_image_bgr_prev1
        curr_image = cv2.cvtColor(next_frame, cv2.COLOR_BGR2RGB)
        t_after_color_conversion=time.time()

        #combined_image_save = 0.1*curr_image + flow_image_bgr

        #CHOOSE IF WE WANT TO TEST WITH ONLY OPTICAL FLOW OR A COMBINATION OF VIDEO AND OPTICAL FLOW
        combined_image = flow_image_bgr
        #combined_image = flow_image_bgr[500:1050, 0:850]

        t_recize= time.time()
        # combined_image = combined_image_save

        combined_image_test = cv2.normalize(combined_image, None, alpha=-1, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        t_normalise= time.time()

        # plt.imshow(combined_image)
        # plt.show()

        #CHOOSE IF WE WANT TO TEST WITH ONLY OPTICAL FLOW OR A COMBINATION OF VIDEO AND OPTICAL FLOW
        # combined_image_test = cv2.resize(combined_image, (0,0), fx=0.5, fy=0.5)
        combined_image_test = cv2.resize(combined_image_test, (0,0), fx=0.5, fy=0.5)

        combined_image_test = combined_image_test.reshape(1, combined_image_test.shape[0], combined_image_test.shape[1], combined_image_test.shape[2])
        t_reshapeandrecize=time.time()

        prediction = model.predict(combined_image_test)
        t_predict=time.time()

        predicted_labels.append(prediction[0][0])
        truth=float(Ground_truth[count])
        file1.writelines(str(prediction[0][0])+ '\n')

        # print(combined_image.shape, np.mean(flow_image_bgr), prediction[0][0])




        cv2.putText(next_frame, str(int(prediction[0][0]-2)), place, font, fontScale,(0,0,255),lineType)
        cv2.putText(next_frame, str(int(truth)), place2, font, fontScale, (0,255,0), lineType)
        cv2.putText(next_frame, 'prediction : ', place3, font, fontScale, (0,0,255), lineType)
        cv2.putText(next_frame, 'Ground truth : ', place4, font, fontScale, (0,255,0), lineType)
        #cv2.putText(combined_image_save, str(prediction[0][0]), place, font, fontScale,fontColor,lineType)
        cv2.imshow('frame',next_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        video_writer.write(next_frame)
       # video_writer_combined.write(combined_image_save.astype('uint8'))

        prev_frame = next_frame
        #flow_image_bgr_prev4 = flow_image_bgr_prev3
        #flow_image_bgr_prev3 = flow_image_bgr_prev2
        #flow_image_bgr_prev2 = flow_image_bgr_prev1
        flow_image_bgr_prev1 = flow_image_bgr_next

        count +=1
        sys.stdout.write('\rprocessed frames: %d of %d' % (count, num_frames))
        t_end=time.time()

        print('total time : ', t_end- t_start)
        print('optical flow conversion time: ' ,t_after_optixcal_conversion-t_start)
        print('color conversion time : ' , t_after_color_conversion-t_after_optixcal_conversion)
        print('recize time : ', t_recize-t_after_color_conversion)
        print(' normalize time : ' , t_normalise-t_recize)
        print(' reshape time : ', t_recize-t_reshapeandrecize)
        print(' predict time : ', t_predict-t_reshapeandrecize)





    t2 = time.time()
    video_reader.release()
    video_writer.release()
    video_writer_combined.release()
    file1.close()
    print(' Prediction completed !')
    print(' Time Taken:', (t2 - t1), 'seconds')

    predicted_labels[0] = predicted_labels[1]
    return predicted_labels

def load_GT(PATH_TRUE_SPEED):
    f = open(PATH_TRUE_SPEED, "r")
    GT=[]
    for x in f:
        date_string = x.split('\n')[0]
        GT.append(date_string)
    return GT


if __name__ == '__main__':

    model = CNNModel()
    model.load_weights(PRE_TRAINED_WEIGHTS)

    print('Testing model...')
    Ground_truth=load_GT(PATH_TRUE_SPEED)
    predicted_labels = predict_from_video(PATH_TEST_VIDEO,Ground_truth,  PATH_TEST_VIDEO_OUTPUT, PATH_COMBINED_TEST_VIDEO_OUTPUT)

    with open(PATH_TEST_LABEL, mode="w") as outfile:
        for label in predicted_labels:
            outfile.write("%s\n" % str(label))
