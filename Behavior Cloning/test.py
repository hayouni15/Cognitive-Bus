from keras.models import load_model

import numpy as np
import cv2
import time

def main():
    print('done')



def img_preprocess(self,img):
    img = img[60:135, :, :]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img / 255
    return img

def predict_angle(self,image):

    image = np.asarray(image)
    image = self.img_preprocess(image)
    image = np.array([image])
    startTime = time.time()
    steering_angle = float(model.predict(image))
    endTime = time.time()
    processing_time=abs(endTime-startTime)
    return steering_angle
        
if __name__ == "__main__":
    model = load_model('/home/ahayouni/Documents/brite-unit2/src/BehaviorCloning/modelVGG.h5')
    model1 = load_model('/home/ahayouni/Documents/brite-unit2/src/BehaviorCloning/modelAlexNet.h5')
    model2 = load_model('modelCNN.h5')
    model3 = load_model('modelInvedia.h5')
    main()
