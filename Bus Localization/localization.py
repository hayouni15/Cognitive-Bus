from typing import List

from ..common.dependencies import *
from ..common import detector, picasso, demosaic
from . import constants
from ..common import common_constants
from ..common.anchor import Anchor
from ..common.database import Database
from ..common.data import data
from ..common.processes import *
from ..common.possible_location import Possible_location_stats

import matplotlib.pyplot as plt


# ----------------------------------------#
# Auther : Abdessattar Hayouni            #
# University of Calgary                   #
# Thales Research and technology          #
###########################################
#       Bus localization                  #
###########################################


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


def get_bearing_error(b1, b2):
    r = (b2 - b1) % 360.0
    # Python modulus has same sign as divisor, which is positive here,
    # so no need to consider negative case
    if r >= 180.0:
        r -= 360.0
    return r


def get_anchor_with_highest_prob(filtered_stats):
    the_anchor = filtered_stats[0]
    for stat in filtered_stats:
        if stat.probability > the_anchor.probability:
            the_anchor = stat
    return the_anchor


def filter_initial_possible_locations(possible_locations, stats_array, heading):
    filtered_stats = None
    for One_anchor in possible_locations:
        nearby_anchors = One_anchor.Load_nearby_anchors(anchor_map, 1)
        distances = One_anchor.calculate_nearby_distances(nearby_anchors)
        bearings = One_anchor.calculate_nearby_bearings(nearby_anchors)
        bearing_error = bearings
        bearing_error = [get_bearing_error(dummy_bearing, float(heading)) for dummy_bearing in bearing_error]
        probability = bearing_error
        probability = [1 - abs(dummy_probability) / 180 for dummy_probability in probability]

        for index in range(len(nearby_anchors)):
            if probability[index] > 0.94:
                stats_array.append(
                    Possible_location_stats(One_anchor, nearby_anchors[index], distances[index],
                                            bearings[index],
                                            bearing_error[index], probability[index]))

    if stats_array:
        filtered_stats = [stats_array[0]]
    for stat in stats_array:
        if not stat.original_anchor_exists(filtered_stats):
            filtered_stats.append(stat)
    for stat in stats_array:
        for index in range(len(filtered_stats)):
            if (stat.original_anchor == filtered_stats[index].original_anchor) and (
                    stat.distance < filtered_stats[index].distance):
                filtered_stats[index] = stat

    return filtered_stats


def update_probability(filtered_stat, stats_array, heading):
    for stat in filtered_stat:
        stat.bearing_error = get_bearing_error(stat.bearing, float(heading))
        stat.probability = 1 - abs(stat.bearing_error / 180)

    return filtered_stat


def get_next_nearest_anchor_in_bearing(anchor, anchor_map, heading):
    radius = 0.01
    distances = []
    while len(distances) == 0:
        radius = radius + 0.01
        near_by_anchors = anchor.Load_nearby_anchors(anchor_map, radius)
        nearby_anchors_filtered = []
        for nearby_anchor in near_by_anchors:
            bearing = anchor.get_bearing_to_another_anchor(nearby_anchor) * 180 / math.pi
            bearing = (360 - ((bearing + 360) % 360))
            if abs(bearing - float(heading)) < 15:
                # near_by_anchors.remove(nearby_anchor)
                nearby_anchors_filtered.append(nearby_anchor)
        # 46.7685628
        # -71.3011732
        distances = anchor.calculate_nearby_distances(nearby_anchors_filtered)
        for d in distances:
            dumb_index = 0
            if d == 0:
                distances.remove(d)
                nearby_anchors_filtered.remove(nearby_anchors_filtered[dumb_index])
            dumb_index += 1
        if distances:
            nearest_anchor_index = distances.index(min(distances))
            if distances[nearest_anchor_index] == 0:
                nearby_anchors_filtered.remove(nearby_anchors_filtered[nearest_anchor_index])
                distances.remove(distances[nearest_anchor_index])
                nearest_anchor_index = distances.index(min(distances))
            nearby_anchor_return = nearby_anchors_filtered[nearest_anchor_index]
        if radius > 0.4:
            nearby_anchor_return = None
            break
    return nearby_anchor_return, radius

    near_by_anchors = anchor.Load_nearby_anchors(anchor_map, 0.35)
    nearby_anchors_filtered = []
    for nearby_anchor in near_by_anchors:
        bearing = anchor.get_bearing_to_another_anchor(nearby_anchor) * 180 / math.pi
        bearing = (360 - ((bearing + 360) % 360))
        if abs(bearing - float(heading)) < 15:
            # near_by_anchors.remove(nearby_anchor)
            nearby_anchors_filtered.append(nearby_anchor)

    distances = anchor.calculate_nearby_distances(nearby_anchors_filtered)
    if distances:
        nearest_anchor_index = distances.index(min(distances))
        if distances[nearest_anchor_index] == 0:
            nearby_anchors_filtered.remove(nearby_anchors_filtered[nearest_anchor_index])
            distances.remove(distances[nearest_anchor_index])
            nearest_anchor_index = distances.index(min(distances))

        return nearby_anchors_filtered[nearest_anchor_index]
    else:
        return None


def remove_duplicates(filtered_stats):
    if filtered_stats:
        new_filtered_stats = [filtered_stats[0]]
        for stat in filtered_stats:
            if not stat.is_duplicate(new_filtered_stats):
                new_filtered_stats.append(stat)
        return new_filtered_stats
    return filtered_stats


def estimate_location_between_anchors(anchor, distance, heading, latitude, longitude):
    R = 6356.8  # radius of earth
    lat = math.radians(anchor.lat)
    lon = math.radians(anchor.lng)
    # brng = math.radians(possible_locations[0].heading)  # use heading saved in database
    brng = math.radians(float(heading))  # use heading from gps

    lat = math.asin(math.sin(lat) * math.cos((distance / 1000) / R) + math.cos(lat) * math.sin(
        (distance / 1000) / R) * math.cos(brng))
    lon += math.atan2(math.sin(brng) * math.sin((distance / 1000) / R) * math.cos(lat),
                      math.cos((distance / 1000) / R) - math.sin(lat) * math.sin(lat))

    lat = math.degrees(lat)
    lon = math.degrees(lon)

    location_estimation_string = str(lat) + '$' + str(lon) + '$' + str(latitude) + '$' + str(longitude)
    db.update_location_estimation(location_estimation_string)
    return lat, lon


def plot_certainty(certainty_array):
    fig = plt.figure('System certainty ')
    plt.suptitle("System certainy(%)")
    plt.plot(certainty_array)
    plt.ylabel('Certainty (%)')
    plt.show(block=False)


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
Data = data(Front_right_cam_file, Side_right_cam_file, Video_timeStamp_file, GPS_timeStamp_file,
            yolo_configuration_file, yoloV3_weights, coco_data)

# ###### db int ###########################
db_host = common_constants.getDatabaseHost()
db_username = common_constants.getDatabaseUsername()
db_password = common_constants.getDatabasePassword()
db_name = common_constants.getDatabaseName()
db = Database(db_host, db_username, db_password, db_name)
db.establish_connection()

# ############ load anchor map ############
anchor_map = db.load_anchor_map(9)
##########################################

# ###### GEt timestamp array from video and gps##
TimeStamp_array = Data.Get_TimeStamp_from_Video()
GPS_time_stamp_array = Data.Get_TimeStamp_from_GPS()
GPS_data_array, time_stamp_array = Data.GPS_data_array_from_string(GPS_time_stamp_array)

# define the known dimensions of the video frames, it is mendatory to know
# these dimensions to successfully read the raw video file
width = constants.width  # float
height = constants.height  # float

# setup video writer
video_writer_file_destination = common_constants.getVideoWriterDestination()
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(
    '/home2/BRITE/' + Data.Front_right_cam_file.split('/')[4].split('.')[0] + '.mp4', fourcc, 20.0,
    (1936, 1216))

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
db.update_possible_locations('')
detection_status = False
certainty_coef = 0.01
certainty_array = [0]
certainty_coef = db.update_certainty_coefficient(certainty_coef)
camera_fps = 20  # 25 frames per second , if real time use FPS which is the time we process each frame
db.update_location_estimation('')
uncertainty_coef = 0
last_confirmed_location = []
estimation_error = []
###########################################################################

# generating the three RGB masks for demosaicking
mask_red, mask_green, mask_blue = demosaic.create_masks(height, width)


def main():
    global frame_count, distance, anchor_distance_frame_count, stopped_count, TL_counter, FH_Counter, TS1_Counter, \
        TS2_Counter, TS3_Counter, anchor_distance, Avg_speed, detection_status, detected, prev_pred, \
        possible_locations, max_pred, last_confirmed_location, certainty_coef, uncertainty_coef, certainty_array
    UI_Active = common_constants.getUIStatus()

    # reading the raw video data and process it

    front_cam = cv2.VideoCapture(Data.Front_right_cam_file)
    right_cam = cv2.VideoCapture(Data.Side_right_cam_file)
    stats_array = []
    filtered_stats = []
    confirmed_location = None
    radius = 0.1
    target_lost = False

    #plt.ion()
    #fig = plt.figure()
    #axes = fig.add_subplot(111)
    #axes.set_autoscale_on(True)
    #axes.autoscale_view(True, True, True)

    lat_est=[]
    lng_est=[]
    lat_gps=[]
    lng_gps=[]

    lat_brite_txt= open('lat_brite.txt', "a+")
    lat_gps_txt = open('lat_gps.txt', "a+")
    lng_brite_txt = open('lng_brite.txt', "a+")
    lng_gps_txt = open('lng_gps.txt', "a+")



    while (front_cam.isOpened()):
        frame_count = frame_count + 1
        TimeStamp = TimeStamp_array[frame_count - 1]
        stime = time.time()
        certainty_array.append(certainty_coef)
        # get gps data corresponding to our timestamp
        [time_stamps, date, latitude, longitude, attitude, velocity, heading] = Data.data_from_time_Stamp(TimeStamp,
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

        if not (frame.size > 0):
            break

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

        ###############################################################
        pred = process_detection(r, r2)
        TL_counter, FH_Counter, TS1_Counter, TS2_Counter, TS3_Counter = assert_detection(TL_counter, FH_Counter,
                                                                                         TS1_Counter,
                                                                                         TS2_Counter,
                                                                                         TS3_Counter, pred)
        current_pred = [TL_counter, FH_Counter, TS1_Counter, TS2_Counter, TS3_Counter]
        anchor_distance_frame_count, Avg_speed, detection_status, detected, distance_from_prev = detection_decision(
            prev_pred,
            current_pred, anchor_distance_frame_count, Avg_speed, frame_d, detection_status, detected,
            anchor_distance)

        if detection_status:

            if not filtered_stats:
                possible_locations = db.initialize_possible_locations(detected)
                filtered_stats = filter_initial_possible_locations(possible_locations, stats_array, heading)

            else:
                new_filtered_stats = []
                for stat in filtered_stats:
                    print(stat.id)
                    if detected == int(stat.nearby_anchor.anchor_type):
                        stat.original_anchor = stat.nearby_anchor
                        stat.nearby_anchor, radius = get_next_nearest_anchor_in_bearing(stat.original_anchor,
                                                                                        anchor_map,
                                                                                        heading)
                        if stat.nearby_anchor:
                            new_filtered_stats.append(stat)
                        else:
                            #  in case no anchor in that direction is not found
                            if len(filtered_stats) == 1:
                                print('no anchor found')
                                certainty_coef = db.update_certainty_coefficient(0.5)
                                last_confirmed_location = stat.original_anchor
                                filtered_stats = None
                            else:
                                filtered_stats.remove(stat)
                    else:
                        if len(filtered_stats) == 1:
                            # handle detection exceptions
                            last_confirmed_location = stat.original_anchor
                            last_id = db.get_max_id()
                            detected_anchor = Anchor(
                                [last_id[0] + 1, detected, est_lat, est_lng, heading, anchor_distance[5], 0])
                            anomaly = stat.original_anchor.detection_anomaly_type(stat.nearby_anchor, detected_anchor)
                            if anomaly == 'new anchor':
                                filtered_stats[0].original_anchor = detected_anchor
                                last_confirmed_location = detected_anchor
                                anchors_in_db = db.load_anchor_map(0)
                                if detected_anchor.anchor_exists_in_db(anchors_in_db):
                                    db.update_weight(detected_anchor.anchor_exists_in_db(anchors_in_db), '+')

                                    print('anchor exists , add to node coefficient')
                                else:
                                    db.save_anchor(detected_anchor.anchor_type, detected_anchor.distance_from_prev,
                                                   detected_anchor.lat, detected_anchor.lng, detected_anchor.heading,
                                                   detected_anchor.weight)
                                print('new anchor')
                            else:
                                print('obstructed anchor')
                                next_anchor, radius = get_next_nearest_anchor_in_bearing(
                                    filtered_stats[0].nearby_anchor, anchor_map, heading)
                                db.update_weight(filtered_stats[0].nearby_anchor, '-')
                                if detected == int(next_anchor.anchor_type):
                                    filtered_stats[0].original_anchor = next_anchor
                                    filtered_stats[0].nearby_anchor, radius = get_next_nearest_anchor_in_bearing(
                                        next_anchor, anchor_map, heading)
                                    new_filtered_stats = filtered_stats

                                    print('anchor obstraction resolved')
                                else:
                                    print('recalibrate')
                                    possible_locations = confirmed_location.recalibrate(anchor_map, 0.4)
                                    filtered_stats = filter_initial_possible_locations(possible_locations, stats_array,
                                                                                       heading)
                                    certainty_coef = 1 / len(possible_locations)
                                    certainty_coef = db.update_certainty_coefficient(certainty_coef)
                                    confirmed_location = None

                                print('estimated location :', est_lat, est_lng)
                filtered_stats = remove_duplicates(new_filtered_stats)

            if len(filtered_stats) == 1:
                print('location confirmed: ', filtered_stats[0].original_anchor.id, radius)
                confirmed_location = filtered_stats[0].original_anchor
            else:
                if len(filtered_stats) > 0:
                    certainty_coef = 1 / len(filtered_stats)
                    certainty_coef = db.update_certainty_coefficient(certainty_coef)


        else:

            if filtered_stats:
                bar_data_probability=[]
                bar_data_id=[]
                for data in filtered_stats:
                    bar_data_probability.append(data.probability)
                    bar_data_id.append(data.original_anchor.id)
                possible_locations_string = ''

                picasso.draw_probibility_bars(frame,bar_data_probability,bar_data_id)

                for stat in filtered_stats:
                    possible_locations_string += str(stat.original_anchor.id) + '$'
                    # update expected anchor

                possible_locations_string += str(radius)
                db.update_possible_locations(possible_locations_string)
                possible_locations = []
                for stat in filtered_stats:
                    possible_locations.append(stat.original_anchor)
                filtered_stats = update_probability(filtered_stats, [], heading)
                most_likely_location = get_anchor_with_highest_prob(filtered_stats)
                print(most_likely_location.original_anchor.id, most_likely_location.probability)
            else:
                if last_confirmed_location:
                    possible_locations_string = ''
                    possible_locations_string += str(last_confirmed_location.id) + '$'
                    db.update_possible_locations(possible_locations_string)
                    stats_array = []
                    possible_locations = [last_confirmed_location]
                    filtered_stats = filter_initial_possible_locations(possible_locations, stats_array, heading)
                    confirmed_location = last_confirmed_location
                    if filtered_stats:
                        last_confirmed_location = None
                    print('after location lost')

        if confirmed_location:
            est_lat, est_lng = estimate_location_between_anchors(confirmed_location, anchor_distance[5] / 2, heading,
                                                                 latitude, longitude)
            lat_est.append(est_lat)
            lng_est.append(est_lng)
            lat_gps.append(latitude)
            lng_est.append(longitude)

            lat_brite_txt.writelines(str(est_lat) + '\n')
            lat_gps_txt.writelines(str(latitude)+ '\n')
            lng_brite_txt.writelines(str(est_lng)+ '\n')
            lng_gps_txt.writelines(str(longitude)+ '\n')

            certainty_coef = db.update_certainty_coefficient(most_likely_location.probability)
            if certainty_coef < 0.65:
                current_location_anchor = Anchor([None, None, est_lat, est_lng, heading, distance_from_prev, 0])

                dumb_nearby_anchor, radius = get_next_nearest_anchor_in_bearing(
                    current_location_anchor,
                    anchor_map,
                    heading)
                if dumb_nearby_anchor:
                    filtered_stats[0].nearby_anchor = dumb_nearby_anchor
                    bearing = current_location_anchor.get_bearing_to_another_anchor(
                        filtered_stats[0].nearby_anchor) * 180 / math.pi
                    bearing = (360 - ((bearing + 360) % 360))
                    prob = 1 - abs((bearing - float(heading)) / 180)
                    filtered_stats[0].probability = prob
                    filtered_stats[0].bearing = bearing

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

        optical_flow = cv2.imread(
            '/home/ahayouni/Documents/brite-unit2/src/speedEstimation/data/brite_reweighted_images_flow/' + str(
                frame_count) + '.jpg')
        optical_flow = cv2.resize(optical_flow, (484, 304))

        combinedFrame[15 + int(height / 4):15 + 2 * int(height / 4),
        width - int(width / 4) - 10:width - 10] = optical_flow

        picasso.draw_route_info(combinedFrame, mvtStatus, TimeStamp, latitude, longitude, attitude, velocity,
                                heading, anchor_distance[5], mvtColor)
        cv2.putText(combinedFrame,'mph :',(constants.width - 100, 400),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0, 250, 0),2,)
        cv2.putText(combinedFrame,str(int(3.6 * float(velocity))), (constants.width - 100, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 250, 0), 2, )

        if UI_Active:
            cv2.imshow('window', combinedFrame)
            plot_certainty(certainty_array)
        writer.write(combinedFrame)
        FPS = 1 / (time.time() - stime)
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        if frame_count == 5999:

            lat_brite_txt.close()
            lat_gps_txt.close()
            lng_brite_txt.close()
            lng_gps_txt.close()
            writer.release()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


# writer.release()


if __name__ == "__main__":
    main()