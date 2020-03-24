import cv2

from ..localization import constants
from ..common.dependencies import *


class risky_objects_count:
    def __init__(self,car_count,bus_count,truck_count,motorcycle_count,person_count,bicycle_count):
        self.car_count=car_count
        self.bus_count=bus_count
        self.truck_count=truck_count
        self.motorcycle_count=motorcycle_count
        self.person_count=person_count
        self.bicycle_count=bicycle_count

def draw_collision_warning_info(frame, risky_objects_count):
    cv2.putText(frame,
                'Objects tracked: ',
                (50 - 20, 68 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                [0, 0, 0],
                2,
                )
    cv2.putText(frame,
                'Persons : ',
                (50 - 20, 200),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                [0, 0, 0],
                2,
                )
    cv2.putText(frame,
                str(risky_objects_count.person_count),
                (230, 200),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                [0, 0, 0],
                2,
                )
    cv2.putText(frame,
                'Cars : ',
                (50 - 20, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                [0, 0, 0],
                2,
                )
    cv2.putText(frame,
                str(risky_objects_count.car_count),
                (230, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                [0, 0, 0],
                2,
                )
    cv2.putText(frame,
                'Trucks : ',
                (50 - 20, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                [0, 0, 0],
                2,
                )
    cv2.putText(frame,
                str(risky_objects_count.truck_count),
                (230, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                [0, 0, 0],
                2,
                )
    cv2.putText(frame,
                'Bus : ',
                (50 - 20, 250),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                [0, 0, 0],
                2,
                )
    cv2.putText(frame,
                str(risky_objects_count.bus_count),
                (230, 250),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                [0, 0, 0],
                2,
                )
    cv2.putText(frame,
                'Motorcycle : ',
                (50 - 20, 300),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                [0, 0, 0],
                2,
                )
    cv2.putText(frame,
                str(risky_objects_count.motorcycle_count),
                (230, 300),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                [0, 0, 0],
                2,
                )
    cv2.putText(frame,
                'Bicycle : ',
                (50 - 20, 350),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                [0, 0, 0],
                2,
                )
    cv2.putText(frame,
                str(risky_objects_count.bicycle_count),
                (230, 350),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                [0, 0, 0],
                2,
                )



def draw_detection_bars(frame,
                        TL_counter,
                        FH_Counter,
                        TS1_Counter,
                        TS2_Counter,
                        TS3_Counter,
                        frame_count,
                        stopped_count,
                        ):
    """
    Simply drawing the detection bars on the top left corner of the screen
    """

    TL = (TL_counter * 10 if TL_counter * 10 < 100 else 97)
    FH = (FH_Counter * 10 if FH_Counter * 10 < 100 else 97)
    TS1 = (TS1_Counter * 10 if TS1_Counter * 10 < 100 else 97)
    TS2 = (TS2_Counter * 10 if TS2_Counter * 10 < 100 else 97)
    TS3 = (TS3_Counter * 10 if TS3_Counter * 10 < 100 else 97)

    # graw background

    cv2.rectangle(frame, (5, 10), (220 - 20, 172 - 38), (240, 255,
                                                         255), -1)
    cv2.rectangle(frame, (7, 12), (218 - 20, 170 - 38), (0, 0, 0), 2)

    # traffic lights bar

    cv2.putText(frame,
                'TL : ',
                (50 - 20, 38),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                [0, 0, 0],
                1,
                )
    cv2.rectangle(frame, (100 - 20, 30), (200 - 20, 38), (0, 0, 0), 2)
    cv2.rectangle(frame, (102 - 20, 32), (int(TL) + 101 - 20, 36), (0,
                                                                    0, 255), -1)

    # fire hydrant bar

    cv2.putText(frame,
                'FH : ',
                (50 - 20, 68 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                [0, 0, 0],
                1,
                )
    cv2.rectangle(frame, (100 - 20, 60 - 10), (200 - 20, 68 - 10), (0,
                                                                    0, 0), 2)
    cv2.rectangle(frame, (102 - 20, 62 - 10), (int(FH) + 101 - 20, 66 -
                                               10), (0, 255, 0), -1)

    # Traffic sign class 1

    cv2.putText(frame,
                'TS1: ',
                (50 - 20, 98 - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                [0, 0, 0],
                1,
                )
    cv2.rectangle(frame, (100 - 20, 90 - 20), (200 - 20, 98 - 20), (0,
                                                                    0, 0), 2)
    cv2.rectangle(frame, (102 - 20, 92 - 20), (int(TS1) + 101 - 20, 96 -
                                               20), (255, 0, 0), -1)

    # traffic sign class2

    cv2.putText(frame,
                'TS2: ',
                (50 - 20, 128 - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                [0, 0, 0],
                1,
                )
    cv2.rectangle(frame, (100 - 20, 120 - 30), (200 - 20, 128 - 30),
                  (0, 0, 0), 2)
    cv2.rectangle(frame, (102 - 20, 122 - 30), (int(TS2) + 101 - 20,
                                                126 - 30), (255, 255, 0), -1)

    # traffic sign class 3

    cv2.putText(frame,
                'TS3: ',
                (50 - 20, 158 - 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                [0, 0, 0],
                1,
                )
    cv2.rectangle(frame, (100 - 20, 150 - 40), (200 - 20, 158 - 40),
                  (0, 0, 0), 2)
    cv2.rectangle(frame, (102 - 20, 152 - 40), (int(TS3) + 101 - 20,
                                                156 - 40), (128, 0, 128), -1)

    # draw track rectangle and start marker

    cv2.rectangle(frame, (5, constants.height - 50), (constants.width -
                                                      5, constants.height - 30), (0, 0, 0), -1)
    cv2.rectangle(frame, (5, constants.height - 50), (constants.width -
                                                      5, constants.height - 30), (250, 250, 250), 1)
    cv2.line(frame, (10, constants.height - 40), (constants.width - 10,
                                                  constants.height - 40), (255, 255, 255), 1)
    cv2.rectangle(frame, (int(constants.width / (frame_count -
                                                 stopped_count + 1)), constants.height - 45),
                  (int(constants.width / (frame_count - stopped_count + 1)) +
                   1, constants.height - 37), (255, 255, 255), 5)


def draw_anchor(current_frame,
                det_record,
                stopped_count,
                frame,
                ):
    """
    this fuction draws the anchors on the buttom bar
    """

    for anchor in det_record:
        draw_at = int(anchor[1] / (current_frame + 1 - stopped_count) *
                      constants.width)
        if anchor[0] == 0:
            cv2.rectangle(frame, (draw_at, constants.height - 40),
                          (draw_at + 1, constants.height - 39), (0, 0,
                                                                 255), 5)
        elif anchor[0] == 1:
            cv2.rectangle(frame, (draw_at, constants.height - 40),
                          (draw_at + 1, constants.height - 39), (0,
                                                                 255, 0), 5)
        elif anchor[0] == 2:
            cv2.rectangle(frame, (draw_at, constants.height - 40),
                          (draw_at + 1, constants.height - 39), (255,
                                                                 0, 0), 5)
        elif anchor[0] == 3:
            cv2.rectangle(frame, (draw_at, constants.height - 40),
                          (draw_at + 1, constants.height - 39), (255,
                                                                 255, 0), 5)
        elif anchor[0] == 4:
            cv2.rectangle(frame, (draw_at, constants.height - 40),
                          (draw_at + 1, constants.height - 39), (128,
                                                                 0, 128), 5)


def draw_route_info(combinedFrame,
                    mvtStatus,
                    TimeStamp,
                    latitude,
                    longitude,
                    attitude,
                    velocity,
                    heading,
                    distance,
                    mvtColor):
    cv2.rectangle(combinedFrame, (constants.width - int(constants.width /
                                                        4) - 10, 10), (constants.width - 10,
                                                                       int(constants.height / 4) + 10), mvtColor,
                  2)
    cv2.rectangle(combinedFrame, (constants.width - 120, 10),
                  (constants.width - 12, 40), mvtColor, -1)
    cv2.putText(combinedFrame,
                'side cam',
                (constants.width - 115, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                2,
                )
    cv2.putText(combinedFrame,
                'Time :',
                (constants.width - 900, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 250, 0),
                2,
                )
    cv2.putText(combinedFrame,
                TimeStamp,
                (constants.width - 805, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 250, 0),
                2,
                )
    cv2.putText(combinedFrame,
                'latitude :',
                (constants.width - 900, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 250, 0),
                2,
                )
    cv2.putText(combinedFrame,
                latitude,
                (constants.width - 790, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 250, 0),
                2,
                )
    cv2.putText(combinedFrame,
                'longitude :',
                (constants.width - 900, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 250, 0),
                2,
                )
    cv2.putText(combinedFrame,
                longitude,
                (constants.width - 785, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 250, 0),
                2,
                )
    cv2.putText(combinedFrame,
                'attitude :',
                (constants.width - 900, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 250, 0),
                2,
                )
    cv2.putText(combinedFrame,
                attitude,
                (constants.width - 795, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 250, 0),
                2,
                )
    cv2.putText(combinedFrame,
                'velocity :',
                (constants.width - 900, 130),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 250, 0),
                2,
                )
    cv2.putText(combinedFrame,
                str(int(3.6*float(velocity))),
                (constants.width - 795, 130),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 250, 0),
                2,
                )
    cv2.putText(combinedFrame,
                'heading :',
                (constants.width - 900, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 250, 0),
                2,
                )
    cv2.putText(combinedFrame,
                heading,
                (constants.width - 795, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 250, 0),
                2,
                )
    cv2.putText(combinedFrame, "distance :", (constants.width - 900, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 250, 0),
                2)
    cv2.putText(combinedFrame, str(int(distance)), (constants.width - 795, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 250, 0),
                2)

def draw_probibility_bars(frame, probabilities,id):
    originY = frame.shape[0]
    frame_height = originY
    originX = 0
    max_index=probabilities.index(min(probabilities))
    for i in range (0,len(probabilities)) :
        if i == max_index:
            color=(0,100,0)
        else:
            color=(0,0,0)
        cv2.rectangle(frame, (originX, originY - 80), (originX + 20, int(frame_height * 0.75*probabilities[i])), color, -1)
        cv2.putText(frame, str(id[i]), (originX, originY - 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
        originX += 30

def draw_top_view(car,frame):
    x=car.track_points[len(car.track_points) - 1].x_center
    y = car.track_points[len(car.track_points) - 1].y_center
    d=car.sum
    width=car.track_points[len(car.track_points) - 1].width
    height=car.track_points[len(car.track_points) - 1].height
    a=-12
    b=17
    c=-6000
    color = (0, 120, 0)

    cv2.rectangle(frame, (int(x), 1000 - 10 * int(d)),
                  (int(x ) + int(40), 1000 - 10 * int(d) - int(80)), color, -1)
    cv2.putText(frame, str(int(car.sum)), (int(  x), 1000 - 10 * int(d)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (250, 0, 0), 2)

def draw_steering_angle(frame,deg):
    x1=300
    y1=100
    length=150

    deg=deg*180/math.pi
    x2= length*math.sin(math.radians(deg))
    y2=length*math.cos(math.radians(deg))

    cv2.line(frame, (x1, 1000-y1), (int(x2+x1),int(1000-y2-y1)), (250, 250, 250), 4)
    #cv2.putText(frame,str(deg))
    cv2.rectangle(frame, (0, 1000),
                  (700,500), (250, 250, 250), 4)


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


