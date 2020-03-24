import os

FRONT_RIGHT_CAM_FILE_ENV = 'FRONT_RIGHT_CAM_FILE'
SIDE_RIGHT_CAM_FILE_ENV = 'SIDE_RIGHT_CAM_FILE'
VIDEO_TIMESTAMP_FILE_ENV = 'VIDEO_TIMESTAMP_FILE'
GPS_TIMESTAMP_FILE_ENV = 'GPS_TIMESTAMP_FILE'
YOLO_CONFIGURATION_FILE_ENV = 'YOLO_CONFIGURATION_FILE'
YOLOV3_WEIGHTS_ENV = 'YOLOV3_WEIGHTS'
COCO_DATA_ENV = 'COCO_DATA'

FRONT_RIGHT_CAM_FILE_COMPRESSED_ENV = 'FRONT_RIGHT_CAM_FILE_COMPRESSED'
SIDE_RIGHT_CAM_FILE_COMPRESSED_ENV = 'SIDE_RIGHT_CAM_FILE_COMPRESSED'
VIDEO_TIMESTAMP_FILE_COMPRESSED_ENV = 'VIDEO_TIMESTAMP_FILE_COMPRESSED'
GPS_TIMESTAMP_FILE_COMPRESSED_ENV = 'GPS_TIMESTAMP_FILE_COMPRESSED'
YOLO_CONFIGURATION_FILE_COMPRESSED_ENV = 'YOLO_CONFIGURATION_FILE_COMPRESSED'
YOLOV3_WEIGHTS_COMPRESSED_ENV = 'YOLOV3_WEIGHTS_COMPRESSED'
COCO_DATA_COMPRESSED_ENV = 'COCO_DATA_COMPRESSED'

VIDEO_WRITER_FILE_DESTINATION_ENV = 'VIDEO_WRITER_FILE_DESTINATION'

MYSQL_DATABASE_HOST_ENV = 'MYSQL_DATABASE_HOST'
MYSQL_DATABASE_USERNAME_ENV = 'MYSQL_DATABASE_USERNAME'
MYSQL_DATABASE_PASSWORD_ENV = 'MYSQL_DATABASE_PASSWORD'
MYSQL_DATABASE_NAME_ENV = 'MYSQL_DATABASE_NAME'
UI_ACTIVE_ENV='UI_ACTIVE'

#raw data
FRONT_RIGHT_CAM_FILE_DEFAULT = '/media/ahayouni/Elements/Trajet1/processed/2019-08-05 13-49-19 Avant droit.raw.pictures.raw'
SIDE_RIGHT_CAM_FILE_DEFAULT = '/media/ahayouni/Elements/Trajet1/processed/2019-08-05 13-49-18 Cote droit.raw.pictures.raw'
VIDEO_TIMESTAMP_FILE_DEFAULT = '/media/ahayouni/Elements/Trajet1/processed/2019-08-05 13-39-19 Avant droit.raw.pictures.timestamps.txt'
GPS_TIMESTAMP_FILE_DEFAULT = '/media/ahayouni/Elements/model-data/newRoute/All-gps-Data.txt'
YOLO_CONFIGURATION_FILE_DEFAULT = '/media/ahayouni/Elements/model-data/cfg/yolov3.cfg'
YOLOV3_WEIGHTS_DEFAULT = '/media/ahayouni/Elements/model-data/yolov3.weights'
COCO_DATA_DEFAULT = '/media/ahayouni/Elements/model-data/cfg/coco.data'

#compressed data
FRONT_RIGHT_CAM_FILE_DEFAULT_COMPRESSED = '/home2/BRITE/simulation_data/2019-08-05 13-39-18 Avant droit.mp4'
#FRONT_RIGHT_CAM_FILE_DEFAULT_COMPRESSED = '/home2/BRITE/simulation_data/bus.mp4'
#FRONT_RIGHT_CAM_FILE_DEFAULT_COMPRESSED = '//home2/BRITE/simulation_data/kitti.mp4'
SIDE_RIGHT_CAM_FILE_DEFAULT_COMPRESSED  = '/home2/BRITE/simulation_data/2019-08-05 13-39-18 Cote droit.mp4'
VIDEO_TIMESTAMP_FILE_DEFAULT_COMPRESSED  = '/home2/BRITE/simulation_data/2019-08-05 13-39-18 Avant droit.raw.pictures.timestamps.txt'
GPS_TIMESTAMP_FILE_DEFAULT_COMPRESSED  = '/home2/BRITE/simulation_data/All-gps-Data.txt'
YOLO_CONFIGURATION_FILE_DEFAULT_COMPRESSED  = '/home2/BRITE/simulation_data/yolov3.cfg'
YOLOV3_WEIGHTS_DEFAULT_COMPRESSED  = '/home2/BRITE/simulation_data/yolov3.weights'
COCO_DATA_DEFAULT_COMPRESSED  = '/home2/BRITE/simulation_data/coco.data'

STEERING_ANGLE  = '/home2/BRITE/simulation_data/steering_angle.txt'


VIDEO_WRITER_FILE_DESTINATION_DEFAULT = '/home2/BRITE/simulation_data/collision_warning.mp4'
MYSQL_DATABASE_HOST_DEFAULT = 'localhost'
MYSQL_DATABASE_USERNAME_DEFAULT = 'root'
MYSQL_DATABASE_PASSWORD_DEFAULT = ''
MYSQL_DATABASE_NAME_DEFAULT = 'BRITE'
UI_ACTIVE_DEFAULT=True

def getUIStatus ():
    if UI_ACTIVE_ENV in os.environ:
        return os.environ[UI_ACTIVE_ENV] == 'True'
    else:
        return UI_ACTIVE_DEFAULT


def getFrontRightCamFile ():
    if FRONT_RIGHT_CAM_FILE_ENV in os.environ:
        return os.environ[FRONT_RIGHT_CAM_FILE_ENV]
    else:
        return FRONT_RIGHT_CAM_FILE_DEFAULT

def getSideRightCamFile ():
    if SIDE_RIGHT_CAM_FILE_ENV in os.environ:
        return os.environ[SIDE_RIGHT_CAM_FILE_ENV]
    else:
        return SIDE_RIGHT_CAM_FILE_DEFAULT

def getGPSTimestampFile():
    if GPS_TIMESTAMP_FILE_ENV in os.environ:
        return os.environ[GPS_TIMESTAMP_FILE_ENV]
    else:
        return GPS_TIMESTAMP_FILE_DEFAULT

def getVideoTimestampFile():
    if VIDEO_TIMESTAMP_FILE_ENV in os.environ:
        return os.environ[VIDEO_TIMESTAMP_FILE_ENV]
    else:
        return VIDEO_TIMESTAMP_FILE_DEFAULT

def getYoloCfgFile():
    if YOLO_CONFIGURATION_FILE_ENV in os.environ:
        return os.environ[YOLO_CONFIGURATION_FILE_ENV]
    else:
        return YOLO_CONFIGURATION_FILE_DEFAULT

def getYoloWeights():
    if YOLOV3_WEIGHTS_ENV in os.environ:
        return os.environ[YOLOV3_WEIGHTS_ENV]
    else:
        return YOLOV3_WEIGHTS_DEFAULT

def getCocoData():
    if COCO_DATA_ENV in os.environ:
        return os.environ[COCO_DATA_ENV]
    else:
        return COCO_DATA_DEFAULT

def getVideoWriterDestination():
    if VIDEO_WRITER_FILE_DESTINATION_ENV in os.environ:
        return os.environ[VIDEO_WRITER_FILE_DESTINATION_ENV]
    else:
        return VIDEO_WRITER_FILE_DESTINATION_DEFAULT

def getDatabaseHost():
    if MYSQL_DATABASE_HOST_ENV in os.environ:
        return os.environ[MYSQL_DATABASE_HOST_ENV]
    else:
        return MYSQL_DATABASE_HOST_DEFAULT

def getDatabaseUsername():
    if MYSQL_DATABASE_USERNAME_ENV in os.environ:
        return os.environ[MYSQL_DATABASE_USERNAME_ENV]
    else:
        return MYSQL_DATABASE_USERNAME_DEFAULT

def getDatabasePassword():
    if MYSQL_DATABASE_PASSWORD_ENV in os.environ:
        return os.environ[MYSQL_DATABASE_PASSWORD_ENV]
    else:
        return MYSQL_DATABASE_PASSWORD_DEFAULT

def getDatabaseName():
    if MYSQL_DATABASE_NAME_ENV in os.environ:
        return os.environ[MYSQL_DATABASE_NAME_ENV]
    else:
        return MYSQL_DATABASE_NAME_DEFAULT


def getFrontRightCamFileCompressed ():
    if FRONT_RIGHT_CAM_FILE_COMPRESSED_ENV in os.environ:
        return os.environ[FRONT_RIGHT_CAM_FILE_COMPRESSED_ENV]
    else:
        return FRONT_RIGHT_CAM_FILE_DEFAULT_COMPRESSED

def getSideRightCamFileCompressed ():
    if SIDE_RIGHT_CAM_FILE_COMPRESSED_ENV in os.environ:
        return os.environ[SIDE_RIGHT_CAM_FILE_COMPRESSED_ENV]
    else:
        return SIDE_RIGHT_CAM_FILE_DEFAULT_COMPRESSED

def getGPSTimestampFileCompressed():
    if GPS_TIMESTAMP_FILE_COMPRESSED_ENV in os.environ:
        return os.environ[GPS_TIMESTAMP_FILE_COMPRESSED_ENV]
    else:
        return GPS_TIMESTAMP_FILE_DEFAULT_COMPRESSED

def getVideoTimestampFileCompressed():
    if VIDEO_TIMESTAMP_FILE_COMPRESSED_ENV in os.environ:
        return os.environ[VIDEO_TIMESTAMP_FILE_COMPRESSED_ENV]
    else:
        return VIDEO_TIMESTAMP_FILE_DEFAULT_COMPRESSED

def getYoloCfgFileCompressed():
    if YOLO_CONFIGURATION_FILE_COMPRESSED_ENV in os.environ:
        return os.environ[YOLO_CONFIGURATION_FILE_COMPRESSED_ENV]
    else:
        return YOLO_CONFIGURATION_FILE_DEFAULT_COMPRESSED

def getYoloWeightsCopmressed():
    if YOLOV3_WEIGHTS_COMPRESSED_ENV in os.environ:
        return os.environ[YOLOV3_WEIGHTS_COMPRESSED_ENV]
    else:
        return YOLOV3_WEIGHTS_DEFAULT_COMPRESSED

def getCocoDataCompressed():
    if COCO_DATA_COMPRESSED_ENV in os.environ:
        return os.environ[COCO_DATA_COMPRESSED_ENV]
    else:
        return COCO_DATA_DEFAULT_COMPRESSED
