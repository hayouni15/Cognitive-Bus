from keras.models import load_model

import numpy as np
import cv2
import time

class steer:
    def __init__(self,model_number):
        self.model_number=model_number
    def load_models(self):





        if self.model_number==1:
            print(' Loading model VGG')
            try:
                #model = load_model('../BehaviorCloning/modelVGG.h5')
                model = load_model('../BehaviorCloning/game.h5')
                self.model = model
                print('VGG model loaded successfully')
            except:
                print("can't find model 1")


        elif self.model_number ==2:
            print(' Loading model modelAlexNet')

            try:
                model = load_model('../BehaviorCloning/modelAlexNet.h5')
            except:
                print("can't find model 2")
            self.model=model
            print('modelAlexNet model loaded successfully')
        elif self.model_number == 3:
            print(' Loading model CNN')

            try:
                model = load_model('../BehaviorCloning/modelCNN.h5')
            except:
                print("can't find model 3")
            self.model = model
            print('CNN model loaded successfully')
        elif self.model_number == 4:
            print(' Loading model Nvedia')

            try:
                model = load_model('../BehaviorCloning/modelInvedia.h5')
            except:
                print("can't find model 4")
            self.model = model
            print('Nvedia model loaded successfully')

    def img_preprocess(self,img):
        #img = img[60:135, :, :]
        img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        img = cv2.GaussianBlur(img, (3, 3), 0)
        img = cv2.resize(img, (200, 66))
        img = img / 255
        return img

    def predict_angle(self,image):

        image = np.asarray(image)
        image = self.img_preprocess(image)
        cv2.imshow('processsed top steering_frame', image)
        image = np.array([image])

        startTime = time.time()
        steering_angle = float(self.model.predict(image))
        endTime = time.time()
        processing_time=abs(endTime-startTime)
        return steering_angle