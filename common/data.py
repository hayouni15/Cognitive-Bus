from ..common.dependencies import *
class data:
    def __init__(self, Front_right_cam_file, Side_right_cam_file, Video_timeStamp_file, GPS_timeStamp_file, yolo_configuration_file, yoloV3_weights, coco_data ):
        self.Front_right_cam_file = Front_right_cam_file
        self.Side_right_cam_file = Side_right_cam_file
        self.Video_timeStamp_file = Video_timeStamp_file
        self.GPS_timeStamp_file = GPS_timeStamp_file
        self.yolo_configuration_file = yolo_configuration_file
        self.yoloV3_weights = yoloV3_weights
        self.coco_data = coco_data

    def Get_TimeStamp_from_GPS(self):
        GPS_time_stamp_array = []
        f = open(self.GPS_timeStamp_file, "r")
        for x in f:
            date_string = x.split('\n')[0]
            GPS_time_stamp_array.append(date_string)
        return GPS_time_stamp_array

    def Get_TimeStamp_from_Video(self):
        Time_stamp_array = []
        f = open(self.Video_timeStamp_file, "r")
        for x in f:
            date_string = x.split('\n')[0]
            Time_stamp_array.append(date_string)
        return Time_stamp_array

    def GPS_data_array_from_string(self, GPS_data):
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

    def data_from_time_Stamp(self, time_Stamp_string, GPS_data_array, time_stamp_array):
        date_time_obj = datetime.datetime.strptime(time_Stamp_string, '%Y-%m-%d %H:%M:%S.%f')
        timestamp = time.mktime(date_time_obj.timetuple())
        arr = np.array(time_stamp_array)
        index = np.where(arr == timestamp)
        result = GPS_data_array[index[0][0]]
        return result