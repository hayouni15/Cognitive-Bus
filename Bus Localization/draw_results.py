import cv2
from matplotlib import pyplot as plt
from PyAstronomy import pyasl
import numpy as np
from common.dependencies import *

def load_GT(PATH_TRUE_SPEED):
    f = open(PATH_TRUE_SPEED, "r")
    GT=[]
    for x in f:
        date_string = float(x.split('\n')[0])
        GT.append(date_string)
    return GT

def get_distance_to_another_anchor(lat1,lng1, lati2, long2):
    R = 6371000  # radius of earth



    delta_lat = math.radians(lati2 - lat1)
    delta_long = math.radians( long2-lng1 )

    lat1 = math.radians(lat1)
    lon1 = math.radians(lng1)
    lat2 = math.radians(lati2)
    lon2 = math.radians(long2)

    haversine = math.sin(delta_lat / 2) * math.sin(delta_lat / 2) + math.cos(lat1) * math.cos(lat2) * math.sin(
        delta_long / 2) * math.sin(delta_long / 2)
    distance_in_between = 2 * R * math.atan2(math.sqrt(haversine), math.sqrt(1 - haversine))*1000-2

    return distance_in_between

if __name__ == '__main__':
    lat_brite =  'lat_brite.txt'
    lat_gps =  'lat_gps.txt'
    lng_brite = 'lng_brite.txt'
    lng_gps = 'lng_gps.txt'


    lat_brite = load_GT(lat_brite)
    lat_gps=load_GT(lat_gps)
    lng_brite = load_GT(lng_brite)
    lng_gps = load_GT(lng_gps)
    skip=2250
    plt.figure(0)
    plt.plot(lat_gps[1:len(lat_brite)-skip],lng_gps[1:len(lat_brite)-skip],color='lightcoral',label='GPS position',linewidth=6)
    plt.plot(lat_brite[1:len(lat_brite)-skip],lng_brite[1:len(lat_brite)-skip],'-b',label='estimsted position',linewidth=2)
    plt.xlabel('Latitude', fontsize=14)
    plt.ylabel('Longitude', fontsize=14)
    plt.legend(loc="upper left")
    plt.show()


    error_vector=[]
    print(len(lat_brite))
    print(len(lat_gps))
    for i in range(1,len(lat_brite)-1000):
        errori=get_distance_to_another_anchor(lat_brite[i],lng_brite[i], lat_gps[i], lng_gps[i])/1000
        error_vector.append(errori)
    plt.figure(1)
    plt.plot(error_vector,'-r',label='Localization error',linewidth=2)
    error_vector = np.array(error_vector)
    sm1 = pyasl.smooth(error_vector, 99 ,'hamming')
    plt.plot(sm1, '-b', label='Average Localization error', linewidth=2)
    plt.xlabel('Frames', fontsize=14)
    plt.ylabel('Location estimation error (m)', fontsize=14)
    plt.legend(loc="upper left")
    plt.show()

    avg=sum(error_vector)/len(error_vector)
    print(avg)
    print(min(error_vector))
    print(max(error_vector))






