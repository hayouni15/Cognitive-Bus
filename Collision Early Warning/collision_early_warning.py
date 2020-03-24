from typing import List

from ..common.dependencies import *
from ..common import detector, picasso, demosaic
from . import constants
from ..common import common_constants
from ..common.anchor import Anchor
from ..common.database import Database
from ..common.picasso import risky_objects_count
from ..common.target import target
from ..common.tracker import tracker

import matplotlib.pyplot as plt


# ----------------------------------------#
# Auther : Abdessattar Hayouni            #
# University of Calgary                   #
# Thales Research and technology          #
###########################################
#       Bus localization                  #
###########################################


def process_detection(pred1, pred2, risky_objects_count):
    """
    and outputs an array of detected elements at each frame without duplication
    """
    pred = []

    for p1 in pred1:
        if (p1[0] == b'car'):
            risky_objects_count.car_count = risky_objects_count.car_count + 1
        elif (p1[0] == b'truck'):
            risky_objects_count.truck_count = risky_objects_count.truck_count + 1
        elif (p1[0] == b'bus'):
            risky_objects_count.bus_count = risky_objects_count.bus_count + 1
        elif (p1[0] == b'motorcycle'):
            risky_objects_count.motorcycle_count = risky_objects_count.motorcycle_count + 1
        elif (p1[0] == b'person'):
            risky_objects_count.person_count = risky_objects_count.person_count + 1
        elif (p1[0] == b'bicycle'):
            risky_objects_count.bicycle_count = risky_objects_count.bicycle_count + 1


def assert_detection(TL_counter, FH_Counter, TS1_Counter, TS2_Counter, TS3_Counter, prediction, ismoving=True):
    """
    this function will keep track of the detection count
    it will increment each anchor count if detection countinues
    and restart counter if detection interrupted
    """
    out = [0.6, 0.7, 0.9, 0.7, 0.7]
    for p in prediction:
        if p[0] == b'traffic light':
            out[0] = 1
            TL_counter += 1.2 * p[1]
        if p[0] == b'fire hydrant':
            out[1] = 1
            FH_Counter += 2.5 * p[1]
        if p[0] == b'stop sign':
            out[2] = 1
            TS1_Counter += 2.9 * p[1]
        if p[0] == b'bench':
            out[3] = 1
            TS2_Counter += 2
        if p[0] == b'parking meter':
            out[4] = 1
            TS3_Counter += 1
    counters = [TL_counter, FH_Counter, TS1_Counter, TS2_Counter, TS3_Counter]
    out = np.multiply(out, counters)
    return out[0], out[1], out[2], out[3], out[4]


def detection_decision(prevpred, pred, anchor_distance_frame_count, Avg_speed, stamp, detection_status, detected,
                       anchor_distance,
                       det_threshhold=10, lost_threshhold=15):
    """
    this function keeps track of the detection count for each anchor type , decides if the threshhold is met ( wether the detection
    is consistent enough to consider the object as an anchor or not ).It returns average speed , detection status ( boolean) and the detected object
    if we have detection
    """
    detection_status = False
    distance_from_prev = anchor_distance[5]
    for i in range(0, 5):
        if prevpred[i] > pred[i] and max_pred[i] > det_threshhold and pred[i] < 0.5:
            max_pred[i] = 0
            if detected_at[i] == 0 or abs(anchor_distance[i]) > lost_threshhold:
                print('>>>>>>>>>>>>>>>>> anchor detected :')
                # in case right comera is needed
                right_camera = stamp
                det_record.append((i, frame_count - stopped_count))
                detected_at[i] = anchor_distance_frame_count[i]
                detection_status = True
                anchor_distance_frame_count[i] = 0
                anchor_distance[i] = 0
                anchor_distance_frame_count[5] = 0
                anchor_distance[5] = 0
                Avg_speed = 0
                detected = i  # return anchor type

            else:
                detected = None  # no detection so no anchor type to return
                print('<<<<<<<<<<<<<<<<< duplication case - anchor ignored : ',
                      abs(frame_count - detected_at[i] - prevpred[i]))
    return anchor_distance_frame_count, Avg_speed, detection_status, detected, distance_from_prev


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
    cv2.imshow('imageB', imageB)


def mse(imageA, imageB):
    # the 'Mean Squared Error' between the /home/ahayouni/anaconda3/bin/condatwo images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = abs(np.sum((imageA.astype("float") - imageB.astype("float")) ** 2))
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err


def max_detection_count(current, max):
    for i in range(0, len(current)):
        if current[i] > max[i]:
            max[i] = current[i]
    return max


def Get_TimeStamp_from_Video(Video_timeStamp_file):
    Time_stamp_array = []
    f = open(Video_timeStamp_file, "r")
    for x in f:
        date_string = x.split('\n')[0]
        Time_stamp_array.append(date_string)
    return Time_stamp_array


def Get_TimeStamp_from_GPS(GPS_timeStamp_file):
    GPS_time_stamp_array = []
    f = open(GPS_timeStamp_file, "r")
    for x in f:
        date_string = x.split('\n')[0]
        GPS_time_stamp_array.append(date_string)
    return GPS_time_stamp_array


def GPS_data_from_string(time_stamp_string):
    data = time_stamp_string.split('$')
    date_time_obj = datetime.datetime.strptime(data[0], '%Y-%m-%d %H:%M:%S.%f')
    timestamp = time.mktime(date_time_obj.timetuple())
    velocity = str(float(data[4]))  # /0.27777778 #to change to km/h
    # print(timestamp)
    return timestamp, data[1], data[2], data[3], data[4], data[5]


def Get_TimeStamp_from_GPS(GPS_timeStamp_file):
    GPS_time_stamp_array = []
    f = open(GPS_timeStamp_file, "r")
    for x in f:
        date_string = x.split('\n')[0]
        GPS_time_stamp_array.append(date_string)
    return GPS_time_stamp_array


def GPS_data_array_from_string(GPS_data):
    GPS_data_array = []
    time_stamp_array = []
    array = []
    for data_line in GPS_data:
        GPS_date = data_line.split('$')[0]
        GPS_latitude = data_line.split('$')[1]
        GPS_longitude = data_line.split('$')[2]
        GPS_attitude = data_line.split('$')[3]
        GPS_velocity = data_line.split('$')[4]
        GPS_heading = data_line.split('$')[5]
        date_time_obj = datetime.datetime.strptime(GPS_date, '%Y-%m-%d %H:%M:%S.%f')
        GPS_timestamp = time.mktime(date_time_obj.timetuple())
        GPS_data_array.append(
            [GPS_timestamp, GPS_date, GPS_latitude, GPS_longitude, GPS_attitude, GPS_velocity, GPS_heading])
        time_stamp_array.append(GPS_timestamp)
    return GPS_data_array, time_stamp_array


def data_from_time_Stamp(time_Stamp_string, GPS_data_array, time_stamp_array):
    global gps_data_index

    date_time_obj = datetime.datetime.strptime(time_Stamp_string, '%Y-%m-%d %H:%M:%S.%f')
    timestamp = time.mktime(date_time_obj.timetuple())
    arr = np.array(time_stamp_array)
    index = np.where(arr == timestamp)
    result = GPS_data_array[index[0][0]]
    return result


def get_next_anchor(anchor_map, anchor_id):
    """
    Knowing the Id of the anchor , get its location in the
    anchor_map array
    """

    def get_indexes(anchor_id, anchor_map):
        return [theAnchorIndex for (theAnchor, theAnchorIndex) in zip(anchor_map, range(len(anchor_map))) if
                theAnchor.id == anchor_id]

        print(get_indexes(anchor_id, anchor_map))

    return get_indexes(anchor_id, anchor_map)[0]


def possible_locations_one_update(possible_locations, anchor_map):
    """
    Updates the possible locations vector by adding the next expected anchor
    """
    new_possible_location = []
    for location in possible_locations:
        lastId = location.id
        try:
            next_anchor_id = get_next_anchor(anchor_map, lastId)
            next_anchor = anchor_map[next_anchor_id + 1]

        except BaseException:
            print('no more anchors')  # return new_possible_location
        new_possible_location.append(next_anchor)
    return new_possible_location


def possible_locations_two_updates(possible_locations, anchor_map, uncertainty_coef):
    """
    In case the detected anchor doesnt match the expected one , we assume that the desired anchor
    was obstructed and therefore we should update the possible locations vector to add the next
    two anchors , the one that we have missed and the one that we are expecting next.
    """
    # Get missed distance and update one time
    # certainty_coef-=0.25
    uncertainty_coef += 0.25
    # save the missed distance
    distance = 10
    # update the possible locations
    possible_locations = possible_locations_one_update(possible_locations, anchor_map)
    # add missed distance

    return possible_locations, uncertainty_coef, distance


def possible_locations_matches(possible_locations, detected, distance, uncertainty_coef, distance_check):
    """
    from all possible location keep only the ones that match the detected pattern,
    pass True to distance_check if you want to use two parameters squence ( anchor type and distance from previous anchor)
    pass False as distance_check to ignore checking distance and only check the nature of anchors
    """
    new_possible_locations = []
    if not distance_check:
        for location in possible_locations:
            if str(detected) == location.anchor_type:
                print('detected matches expected: ', location)
                new_possible_locations.append(location)
    else:
        for location in possible_locations:
            distance_truth = location.distance_from_prev
            if str(detected) == location.anchor_type and abs(float(distance_truth) - distance) < 15:
                print('detected matches expected: ', location)
                new_possible_locations.append(location)
                uncertainty_coef = 0

    return new_possible_locations, uncertainty_coef


def get_distance_between_two_anchors(lati1, long1, lati2, long2):
    R = 6356.8  # radius of earth
    lat1 = math.radians(lati1)
    lon1 = math.radians(long1)
    lat2 = math.radians(lati2)
    lon2 = math.radians(long2)

    delta_lat = math.radians(lati2 - lati1)
    delta_long = math.radians(long1 - long2)

    haversine = math.sin(delta_lat / 2) * math.sin(delta_lat / 2) + math.cos(lat1) * math.cos(lat2) * math.sin(
        delta_long / 2) * math.sin(delta_long / 2)
    distance_in_between = 2 * R * math.atan2(math.sqrt(haversine), math.sqrt(1 - haversine))

    return distance_in_between


def remove_past_anchors(All_locations, last_confirmed_location):
    new_locations_Vector = []
    for location in All_locations:
        if location.id > last_confirmed_location.id:
            new_locations_Vector.append(location)
    return new_locations_Vector


# ####### Define Route and segment ########
route = 801
segment = 5
latitude = 47.8
longitude = -71.2
FPS = 2
###########################################
# ###### Files location ##############

Front_right_cam_file = common_constants.getFrontRightCamFileCompressed()
Side_right_cam_file = common_constants.getSideRightCamFileCompressed()
Video_timeStamp_file = common_constants.getVideoTimestampFileCompressed()
GPS_timeStamp_file = common_constants.getGPSTimestampFileCompressed()

yolo_configuration_file = common_constants.getYoloCfgFileCompressed()
yoloV3_weights = common_constants.getYoloWeightsCopmressed()
coco_data = common_constants.getCocoDataCompressed()

video_writer_file_destination = common_constants.getVideoWriterDestination()

# ###### db int ###########################
db_host = common_constants.getDatabaseHost()
db_username = common_constants.getDatabaseUsername()
db_password = common_constants.getDatabasePassword()
db_name = common_constants.getDatabaseName()
db = Database(db_host, db_username, db_password, db_name)
# db.establish_connection()

# ############ load anchor map ############
# anchor_map = db.load_anchor_map()
# anchor_map = db.load_anchor_map()
##########################################

# ###### GEt timestamp array from video and gps##
TimeStamp_array = Get_TimeStamp_from_Video(Video_timeStamp_file)
GPS_time_stamp_array = Get_TimeStamp_from_GPS(GPS_timeStamp_file)
GPS_data_array, time_stamp_array = GPS_data_array_from_string(GPS_time_stamp_array)

# define the known dimensions of the video frames, it is mendatory to know
# these dimensions to successfully read the raw video file
width = constants.width  # float
height = constants.height  # float

# setup video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# writer = cv2.VideoWriter(video_writer_file_destination, fourcc, 20.0, (width, height))

frame_count = 0
stopped_count = 0
# anchor_distnace_frame_count[5] keeps track of the distance of the
# previous anchor( any type)
anchor_distance_frame_count = [0, 0, 0, 0, 0, 0]
anchor_distance = [0, 0, 0, 0, 0, 0]
# define the anchors' counters
TL_counter = 0
FH_Counter = 0
TS1_Counter = 0
TS2_Counter = 0
TS3_Counter = 0
# prev pred will keep track of the predictions for the previous frame
prev_pred = [TL_counter, FH_Counter, TS1_Counter, TS2_Counter, TS3_Counter]
# max_pred will keep track of the maximum number of detection
max_pred = np.zeros_like(prev_pred)
detected_at = [0, 0, 0, 0, 0]
det_record = []

# setup the net , load the configuration file and the weights
net = detector.load_net(bytes(yolo_configuration_file, encoding='utf-8'), bytes(yoloV3_weights, encoding='utf-8'),
                        0)
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
gps_data_index = 0
# ################### Localization parameter initialization ################
confirmed_location = []
possible_locations = []
detected = None
expected_anchor = []
distance = 0
# .update_possible_locations('')
detection_status = False
certainty_coef = 0.01
certainty_array = [0]
# certainty_coef = db.update_certainty_coefficient(certainty_coef)
camera_fps = 20  # 25 frames per second , if real time use FPS which is the time we process each frame
# db.update_location_estimation('')
uncertainty_coef = 0
last_confirmed_location = []
track_lost = False
estimation_error = []
###########################################################################

# generating the three RGB masks for demosaicking
mask_red, mask_green, mask_blue = demosaic.create_masks(height, width)
risky_objects_count = risky_objects_count(0, 0, 0, 0, 0, 0)


def main():
    global frame_count, distance, anchor_distance_frame_count, stopped_count, TL_counter, FH_Counter, TS1_Counter, \
        TS2_Counter, TS3_Counter, anchor_distance, Avg_speed, detection_status, detected, prev_pred, \
        possible_locations, max_pred, last_confirmed_location, certainty_coef, uncertainty_coef, certainty_array
    UI_Active = common_constants.getUIStatus()

    # reading the raw video data and process it

    front_cam = cv2.VideoCapture(Front_right_cam_file)
    right_cam = cv2.VideoCapture(Side_right_cam_file)
    cars_track_vector = []
    while (front_cam.isOpened()):

        risky_objects_count.car_count = 0
        risky_objects_count.bicycle_count = 0
        risky_objects_count.person_count = 0
        risky_objects_count.motorcycle_count = 0
        risky_objects_count.bus_count = 0
        risky_objects_count.truck_count = 0
        frame_count = frame_count + 1
        TimeStamp = TimeStamp_array[frame_count - 1]
        stime = time.time()
        certainty_array.append(certainty_coef)
        # get gps data corresponding to our timestamp
        [time_stamps, date, latitude, longitude, attitude, velocity, heading] = data_from_time_Stamp(TimeStamp,
                                                                                                     GPS_data_array,
                                                                                                     time_stamp_array)
        # print(time.time() - stime)
        # print('time stamp :',TimeStamp)
        # update distance crossed since last detection
        distance += float(velocity) / camera_fps
        # print('time stamp :',TimeStamp)
        for indx in range(0, 6):
            anchor_distance_frame_count[indx] += 1
            anchor_distance[indx] += float(velocity) / 20
        # anchor_distance_frame_count+=1

        ret, frame = front_cam.read()
        ret_d, frame_d = right_cam.read()

        # initialize first frame for movement detection
        xtime = time.time()
        # print('frame reading time :', xtime - stime)
        if (frame_count == 1):
            oldFrame = frame

        ############################################################
        # ###########   < movement detection > #####################
        ############################################################

        # old framne and new frame conversion to gray scale
        Frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        OldFrame = cv2.cvtColor(oldFrame, cv2.COLOR_BGR2GRAY)
        ssim_time = time.time()
        # compare the two images to detect behicle movement
        compare_images(Frame[0:int(0.45 * height), 0:int(width)], OldFrame[0:int(0.45 * height), 0:int(width)])
        # print('ssim process time :', time.time() - ssim_time)
        oldFrame = frame.copy()
        # use the moving average with a window of 10 to avoid sudden
        # changes in the SSI
        if frame_count > 10:
            window = frame_count % 10 + 1
            startingIndex = frame_count - window
            length = len(ssim_array[startingIndex:frame_count])
            avg = sum(ssim_array[startingIndex:frame_count]) / window
            S_avg.append(avg)
        else:
            S_avg.append(ssim_array[frame_count - 1])
        # if the structural similarity index is greater than the
        # threshold 0.93 we are moving
        if (S_avg[frame_count - 1] > 0.97):
            for indx in range(0, 6):
                anchor_distance_frame_count[indx] -= 1
                anchor_distance[indx] -= float(velocity) / 20

            # anchor_distance_frame_count-=1
            stopped_count += 1
            mvtStatus = 'stopped'
            mvtColor = [0, 0, 250]  # print('stopped')
        # if the structural similarity index doesn't reach the
        # threshhold the vehicle is stopped
        else:
            for indx in range(0, 6):
                anchor_distance_frame_count[indx] += 1
                anchor_distance[indx] += float(velocity) / 20
            # anchor_distance_frame_count+=0 # increment the distance
            # count only when the vehicle is moving
            Avg_speed += float(velocity)
            mvtStatus = 'Moving'
            mvtColor = [0, 250, 0]  # print('Moving')

        #############################################################
        cv2.putText(frame_d, "SSIM :", (width - 500, height - 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 4)
        cv2.putText(frame_d, str(S_avg[frame_count - 1]), (width - 300, height - 100), cv2.FONT_HERSHEY_SIMPLEX,
                    2, [0, 2, 0], 4)
        ##############################################################
        # ################### objects detection and mapping ##########
        ##############################################################
        det_start_time = time.time()
        half_size_frame_front = frame.copy()
        half_size_frame_side = frame_d.copy()
        half_size_frame_front = cv2.resize(half_size_frame_front, (int(width / 3), int(height / 3)))
        half_size_frame_side = cv2.resize(half_size_frame_side, (int(width / 3), int(height / 3)))
        # detect object on the front camera
        r = detector.detect(net, meta, half_size_frame_front)
        # detect objects on the right camera
        r2 = detector.detect(net, meta, half_size_frame_side)

        # first detector using the coco dataset and the pretrained yolo
        # weights( fire hydrants , traffic lights and stop signs)
        for i in r:
            x, y, w, h = i[2][0], i[2][1], i[2][2], i[2][3]
            xmin, ymin, xmax, ymax = detector.convertBack(float(x), float(y), float(w), float(h))
            pt1 = (3 * xmin, 3 * ymin)
            pt2 = (3 * xmax, 3 * ymax)

            if (i[0] == b'car'):
                counter = 0
                cv2.rectangle(frame, pt1, pt2, (0, 0, 255), 5)
                target_car = target(x, y, w, h)
                if not cars_track_vector:
                    car_tracker = tracker([target_car], 0,0)
                    cars_track_vector.append(car_tracker)

                else:
                    for car in cars_track_vector:
                        car.tracked_count=car.tracked_count+1
                        if target_car.exists(car.track_points[len(car.track_points)-1]):
                            car.track_points.append(target_car)
                            car.track_count = car.track_count + 1
                            print('exists', (x, y))
                        else:
                            counter += 1

                    if counter == len(cars_track_vector):
                        car.track_count = car.track_count + 1
                        car_tracker = tracker([target_car], 0,0)
                        cars_track_vector.append(car_tracker)
                        print('new', (x, y))
            elif (i[0] == b'truck'):
                cv2.rectangle(frame, pt1, pt2, (0, 0, 255), 5)
            elif (i[0] == b'bus'):
                cv2.rectangle(frame, pt1, pt2, (0, 0, 255), 5)
            elif (i[0] == b'motorcycle'):
                cv2.rectangle(frame, pt1, pt2, (0, 0, 255), 5)
            elif (i[0] == b'person'):
                cv2.rectangle(frame, pt1, pt2, (255, 0, 0), 5)
            elif (i[0] == b'bicycle'):
                cv2.rectangle(frame, pt1, pt2, (255, 0, 0), 5)
            else:
                cv2.rectangle(frame, pt1, pt2, (0, 255, 0), 1)
            cv2.putText(frame, i[0].decode() + " [" + str(round(i[1] * 100, 2)) + "]" + "[" + str(int(x)) + "," + str(
                int(y)) + "]", (pt1[0], pt1[1] + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, [0, 255, 0], 1)
        print('tracker vector length', len(cars_track_vector))
        # draw the tracking vectors
        for car in cars_track_vector:
            if car.track_count > len(car.track_points) + 5:
                cars_track_vector.remove(car)
            if len(car.track_points) > 2:
                car.draw_track(frame)
                car.plot_regression_vector(frame)
                car.plot_extended_line(frame)
                # cv2.arrowedLine(frame, (3 * int(car.track_points[len(car.track_points) - 1].x_center),
                #                         3 * int(car.track_points[len(car.track_points) - 1].y_center)), (
                #                 3 * int(car.track_points[len(car.track_points) - 2].x_center),
                #                 int(3 * car.track_points[len(car.track_points) - 2].y_center)), (0, 0, 255), 5)
        # second detector using the customized dataset (traffic signs )
        for i in r2:
            x, y, w, h = i[2][0], i[2][1], i[2][2], i[2][3]
            xmin, ymin, xmax, ymax = detector.convertBack(float(x), float(y), float(w), float(h))
            pt1 = (3 * xmin, 3 * ymin)
            pt2 = (3 * xmax, 3 * ymax)
            if (i[0] == b'car'):
                cv2.rectangle(frame_d, pt1, pt2, (0, 0, 255), 5)
            elif (i[0] == b'truck'):
                cv2.rectangle(frame_d, pt1, pt2, (0, 0, 255), 5)
            elif (i[0] == b'bus'):
                cv2.rectangle(frame_d, pt1, pt2, (0, 0, 255), 5)
            elif (i[0] == b'motorcycle'):
                cv2.rectangle(frame_d, pt1, pt2, (0, 0, 255), 5)
            elif (i[0] == b'person'):
                cv2.rectangle(frame_d, pt1, pt2, (255, 0, 0), 5)
            elif (i[0] == b'bicycle'):
                cv2.rectangle(frame_d, pt1, pt2, (255, 0, 0), 5)
            else:
                cv2.rectangle(frame_d, pt1, pt2, (0, 255, 0), 1)
            cv2.putText(frame_d, i[0].decode() + " [" + str(round(i[1] * 100, 2)) + "]", (pt1[0], pt1[1] + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, [0, 255, 0], 2)

        ###############################################################
        # pred = process_detection(r, r2)
        process_detection(r, r2, risky_objects_count)
        # draw collision circle
        cv2.circle(frame, (400, 1250), 100, (0, 0, 250), 2)
        cv2.circle(frame, (400, 1250), 200, (0, 204, 204), 2)
        cv2.circle(frame, (400, 1250), 300, (0, 250, 0), 2)

        picasso.draw_collision_warning_info(frame, risky_objects_count)
        dettime = time.time()
        # print('detection time :', dettime - det_start_time)
        TL_counter, FH_Counter, TS1_Counter, TS2_Counter, TS3_Counter = assert_detection(TL_counter, FH_Counter,
                                                                                         TS1_Counter,
                                                                                         TS2_Counter,
                                                                                         TS3_Counter, [])
        current_pred = [TL_counter, FH_Counter, TS1_Counter, TS2_Counter, TS3_Counter]
        anchor_distance_frame_count, Avg_speed, detection_status, detected, distance_from_prev = detection_decision(
            prev_pred,
            current_pred, anchor_distance_frame_count, Avg_speed, frame_d, detection_status, detected,
            anchor_distance)

        prev_pred = current_pred
        max_pred = max_detection_count(current_pred, max_pred)
        print('current frame : ', frame_count)
        # picasso.draw_detection_bars(frame, TL_counter, FH_Counter, TS1_Counter, TS2_Counter, TS3_Counter,
        #  frame_count, stopped_count)
        picasso.draw_anchor(frame_count, det_record, stopped_count, frame)
        if UI_Active:
            cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty("window", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        combinedFrame = frame.copy()
        imS = cv2.resize(frame_d, (int(width / 4), int(height / 4)))
        combinedFrame[10:10 + int(height / 4), width - int(width / 4) - 10:width - 10] = imS

        picasso.draw_route_info(combinedFrame, mvtStatus, TimeStamp, latitude, longitude, attitude, velocity,
                                heading, anchor_distance[5], mvtColor)

        if UI_Active:
            cv2.imshow('window', combinedFrame)
        # writer.write(combinedFrame)
        # print('overall processing time: ', time.time() - stime)
        FPS = 1 / (time.time() - stime)
        # print('distance ', anchor_distance)
        # print('distance ', distance)
        # print('FPS {:.1f}'.format(1 / (time.time() - stime)))
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # writer.release()


if __name__ == "__main__":
    main()
