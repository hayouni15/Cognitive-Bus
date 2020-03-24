from typing import List
from ..common.dependencies import *
from ..common import detector, picasso, demosaic
from . import constants
from ..common import common_constants
from ..common.anchor import Anchor
from ..common.database import Database
from ..common.data import data
from ..common.processes import *

###########################################
# Auther : Abdessattar Hayouni            #
# University of Calgary                   #
# Thales Research and technology          #
###########################################
#       Anchor map creation               #
# Input :- Raw video file of the route to #
#          map                            #
#        - txt file of the GPS data       #
#        - txt file of the video timestamp#
#          to associate anchors to GPS    #
#          location                       #
# Output: database of anchors:            #
#                       - latitude        #
#                       - longitude       #
#                       - altitude        #
#                       - velocity        #
#                       - heading         #
###########################################

def anchor(prevpred, pred, anchor_distance_frame_count, Avg_speed, stamp, det_threshhold=10, lost_threshhold=20):
    for i in range(0, 5):

        if prevpred[i] > pred[i] and max_pred[i] > det_threshhold and pred[i] < 0.5:
            max_pred[i] = 0
            if detected_at[i] == 0 or abs(anchor_distance_frame_count[i]) > lost_threshhold:
                print('>>>>>>>>>>>>>>>>> anchor added :')
                det_record.append((i, frame_count - stopped_count))
                detected_at[i] = anchor_distance_frame_count[i]
                db.save_anchor(str(i), Avg_speed / VIDEO_FPS, latitude, longitude, heading, 0)
                anchor_distance_frame_count[i] = 0
                anchor_distance_frame_count[5] = 0
                Avg_speed = 0
            else:
                print('<<<<<<<<<<<<<<<<< same anchor ( not added) : ', abs(frame_count - detected_at[i] - prevpred[i]))
    return anchor_distance_frame_count, Avg_speed

def compare_images(imageA, imageB):
    """
    This function will compute the mse and the Structural similarity index
    which will be used to decide if the car is moving or stationary
    """
    hight, width = imageA.shape
    imageA = cv2.resize(imageA, (int(width / 20), int(hight / 20)))
    imageB = cv2.resize(imageB, (int(width / 20), int(hight / 20)))
    s = measure.compare_ssim(imageA, imageB)
    ssim_array.append(s)
    cv2.imshow('imageA', imageA)

# ####### Define Route and segment ########
route = 801
segment = 1
latitude = 47.8
longitude = -71.2
FPS=0
VIDEO_FPS = 25
# ##########################################
# ###### Define file location ##############
Front_right_cam_file = common_constants.getFrontRightCamFileCompressed()
Side_right_cam_file = common_constants.getSideRightCamFileCompressed()
Video_timeStamp_file = common_constants.getVideoTimestampFileCompressed()
GPS_timeStamp_file = common_constants.getGPSTimestampFileCompressed()
yolo_configuration_file = common_constants.getYoloCfgFileCompressed()
yoloV3_weights = common_constants.getYoloWeightsCopmressed()
coco_data = common_constants.getCocoDataCompressed()
Data = data(Front_right_cam_file, Side_right_cam_file, Video_timeStamp_file, GPS_timeStamp_file,
            yolo_configuration_file, yoloV3_weights, coco_data)
# ###### db init ###########################
db_host = common_constants.getDatabaseHost()
db_username = common_constants.getDatabaseUsername()
db_password = common_constants.getDatabasePassword()
db_name = common_constants.getDatabaseName()
db = Database(db_host, db_username, db_password, db_name)
db.establish_connection()
####### GEt time stamp array from video ##
TimeStamp_array = Data.Get_TimeStamp_from_Video()
GPS_time_stamp_array = Data.Get_TimeStamp_from_GPS()
GPS_data_array, time_stamp_array = Data.GPS_data_array_from_string(GPS_time_stamp_array)
# setup video writer
video_writer_file_destination = common_constants.getVideoWriterDestination()
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(
    '/home2/BRITE/' + Data.Front_right_cam_file.split('/')[4].split('.')[0] + '.mp4', fourcc, 20.0,
    (1936, 1216))

# frame count
frame_count = 0
stopped_count = 0
anchor_distance_frame_count = [0, 0, 0, 0, 0,
                               0]  # anchor_distnace_frame_count[5] keeps track of the distance of the previous anchor( any type)
# define the anchors' counters
TL_counter = 0
FH_Counter = 0
TS1_Counter = 0
TS2_Counter = 0
TS3_Counter = 0
prev_pred = [TL_counter, FH_Counter, TS1_Counter, TS2_Counter,
             TS3_Counter]  # prev pred will keep track of the predictions for the previous frame
max_pred = np.zeros_like(prev_pred)  # max_pred will keep track of the maximum number of detection
detected_at = [0, 0, 0, 0, 0]
det_record = []
# read video and start detection use cv2.videocapture(0) if using comera feed instead
width = 1936  # float
height = 1216  # float
# setup the net , load the configuration file and the weights
net = detector.load_net(bytes(yolo_configuration_file, encoding='utf-8'), bytes(yoloV3_weights, encoding='utf-8'), 0)
meta = detector.load_meta(bytes(coco_data, encoding='utf-8'))
# initialize movement detection parameters
ssim_array = []  # structural similarity index array
mse_array = []  # MSE index array
S_avg = []  # SSIM moving average array
s = 0  # structural similarity index
m = 0  # MSE index
avg = 0
Avg_speed = 0
mvtStatus = 'stopped'
mvtColor = [0, 0, 250]
# generating the three RGB masks for demosaicking
mask_red, mask_green, mask_blue = demosaic.create_masks(1216, 1936)
threads = []


def main():
    global frame_count, distance, anchor_distance_frame_count, stopped_count, TL_counter, FH_Counter, TS1_Counter, \
        TS2_Counter, TS3_Counter, anchor_distance_frame_count, Avg_speed, detection_status, detected, prev_pred, \
        possible_locations, max_pred, last_confirmed_location, certainty_coef, uncertainty_coef, frame, heading, Avg_speed, FPS, latitude, longitude
    UI_Active = common_constants.getUIStatus()
    colors = [tuple(255 * np.random.rand(3)) for i in range(20)]
    front_cam = cv2.VideoCapture(Data.Front_right_cam_file)
    right_cam = cv2.VideoCapture(Data.Side_right_cam_file)

    while (front_cam.isOpened()):
        time_check1 = time.time()
        ret, frame = front_cam.read()
        ret_d, frame_d = right_cam.read()
        time_check2 = time.time()
        print('reading videos :', time_check2 - time_check1)
        frame_count += 1
        TimeStamp = TimeStamp_array[frame_count - 1]
        # get gps data corresponding to our timestamp
        [time_stamps, date, latitude, longitude, attitude, velocity, heading] = Data.data_from_time_Stamp(TimeStamp,
                                                                                                     GPS_data_array,
                                                                                                     time_stamp_array)
        time_check3 = time.time()
        print('reading time stamp :', time_check3 - time_check2)
        for indx in range(0, 6):
            anchor_distance_frame_count[indx] += 1

        # initialize first frame for movement detection (Structural similarity index)
        if (frame_count == 1):
            oldFrame = frame

        ############################################################
        ############   < movement detection > ########################
        ############################################################
        time_check4 = time.time()
        # old framne and new frame conversion to gray scale
        Frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        OldFrame = cv2.cvtColor(oldFrame, cv2.COLOR_BGR2GRAY)
        time_check5 = time.time()
        print('cvt frame time ', time_check5 - time_check4)
        compare_images(Frame[0:int(0.45 * height), 0:int(width)], OldFrame[0:int(0.45 * height), 0:int(width)])
        oldFrame = frame
        if frame_count > 10:
            window = frame_count % 10 + 1
            startingIndex = frame_count - window
            length = len(ssim_array[startingIndex:frame_count])
            avg = sum(ssim_array[startingIndex:frame_count]) / window
            S_avg.append(avg)
        else:
            S_avg.append(ssim_array[frame_count - 1])
        if (S_avg[frame_count - 1] > 0.97):
            for indx in range(0, 6):
                anchor_distance_frame_count[indx] -= 1

            # anchor_distance_frame_count-=1
            stopped_count += 1
            mvtStatus = 'stopped'
            mvtColor = [0, 0, 250]

        else:

            for indx in range(0, 6):
                anchor_distance_frame_count[indx] += 1
            Avg_speed += float(velocity)
            mvtStatus = 'Moving'
            mvtColor = [0, 250, 0]
            # print('Moving')
            time_check6 = time.time()
            print('mvt det ', time_check6 - time_check5)
        #############################################################

        ##############################################################
        #################### objects detection and mapping ###########
        ##############################################################
        detstart_time = time.time()
        half_size_frame_front = frame.copy()
        half_size_frame_side = frame_d.copy()
        half_size_frame_front = cv2.resize(half_size_frame_front, (int(width / 3), int(height / 3)))
        half_size_frame_side = cv2.resize(half_size_frame_side, (int(width / 3), int(height / 3)))
        r = detector.detect(net, meta, half_size_frame_front)
        r2 = detector.detect(net, meta, half_size_frame_side)
        # first detector using the coco dataset and the pretrained yolo weights( fire hydrants and traffic lights)
        for i in r:
            x, y, w, h = i[2][0], i[2][1], i[2][2], i[2][3]
            xmin, ymin, xmax, ymax = detector.convertBack(float(x), float(y), float(w), float(h))
            pt1 = (3 * xmin, 3 * ymin)
            pt2 = (3 * xmax, 3 * ymax)
            if (i[0] == b'traffic light'):
                cv2.rectangle(frame, pt1, pt2, (0, 0, 255), 5)
            elif (i[0] == b'fire hydrant'):
                cv2.rectangle(frame, pt1, pt2, (0, 255, 0), 5)
            elif (i[0] == b'stop sign'):
                cv2.rectangle(frame, pt1, pt2, (255, 0, 0), 5)
            else:
                cv2.rectangle(frame, pt1, pt2, (0, 255, 0), 1)
            cv2.putText(frame, i[0].decode() + " [" + str(round(i[1] * 100, 2)) + "]", (pt1[0], pt1[1] + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, [0, 255, 0], 1)
        # second detector using the customized dataset (traffic signs )
        for i in r2:
            x, y, w, h = i[2][0], i[2][1], i[2][2], i[2][3]
            xmin, ymin, xmax, ymax = detector.convertBack(float(x), float(y), float(w), float(h))
            pt1 = (3 * xmin, 3 * ymin)
            pt2 = (3 * xmax, 3 * ymax)
            if (i[0] == b'traffic light'):
                cv2.rectangle(frame_d, pt1, pt2, (0, 0, 255), 5)
            elif (i[0] == b'fire hydrant'):
                cv2.rectangle(frame_d, pt1, pt2, (0, 255, 0), 5)
            elif (i[0] == b'stop sign'):
                cv2.rectangle(frame_d, pt1, pt2, (255, 0, 0), 5)
            else:
                cv2.rectangle(frame_d, pt1, pt2, (0, 255, 0), 3)
            cv2.putText(frame_d, i[0].decode() + " [" + str(round(i[1] * 100, 2)) + "]", (pt1[0], pt1[1] + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, [0, 255, 0], 2)
        print('detection time : ', time.time() - detstart_time)
        ###############################################################
        pred = process_prediction(r, r2)
        print('combined pred vector :', pred)
        TL_counter, FH_Counter, TS1_Counter, TS2_Counter, TS3_Counter = assert_detection(TL_counter, FH_Counter,
                                                                                         TS1_Counter,
                                                                                         TS2_Counter,
                                                                                         TS3_Counter, pred)
        current_pred = [TL_counter, FH_Counter, TS1_Counter, TS2_Counter, TS3_Counter]
        anchor_distance_frame_count, Avg_speed = anchor(prev_pred, current_pred, anchor_distance_frame_count,
                                                        Avg_speed, frame_d)
        prev_pred = current_pred
        max_pred = max_detection_count(current_pred, max_pred)
        print('current frame : ', frame_count)

        picasso.draw_detection_bars(frame, TL_counter, FH_Counter, TS1_Counter, TS2_Counter, TS3_Counter,
                                    frame_count, stopped_count)
        picasso.draw_anchor(frame_count, det_record, stopped_count, frame)
        if UI_Active:
            cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty("window", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        combinedFrame = frame.copy()
        imS = cv2.resize(frame_d, (int(width / 4), int(height / 4)))
        combinedFrame[10:10 + int(height / 4), width - int(width / 4) - 10:width - 10] = imS

        picasso.draw_route_info(combinedFrame, mvtStatus, TimeStamp, latitude, longitude, attitude, velocity,
                                heading, Avg_speed / VIDEO_FPS, mvtColor)
        cv2.imshow('window', combinedFrame)
        writer.write(combinedFrame)
        FPS = 1 / (time.time() - time_check1)
        print('distance ', Avg_speed / VIDEO_FPS)

        print('FPS {:.1f}'.format(1 / (time.time() - time_check1)))
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        if cv2.waitKey(1) & 0xFF == ord('q'):
            writer.release()
            break

    front_cam.release()
    right_cam.release()
    cv2.destroyAllWindows()
    writer.release()


if __name__ == "__main__":
    main()
