# from model2 import CNNModel
from model import CNNModel
import cv2
import numpy as np
import os, sys
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.callbacks import EarlyStopping, ModelCheckpoint

from imgaug import augmenters as iaa

PATH_DATA_FOLDER = './data/'
PATH_TRAIN_LABEL = '/home/ahayouni/Documents/brite-unit2/src/BehaviorCloning/steering_ground_truth.txt'
PATH_TRAIN_IMAGES_FOLDER =  '//data/train_images/'


TYPE_FLOW_PRECOMPUTED = 0
TYPE_ORIGINAL = 1

BATCH_SIZE = 50
EPOCH = 150

MODEL_NAME = 'brite_steering'
# MODEL_NAME = 'CNNModel_combined'

def load_labels(labels_path):
    f = open(labels_path, "r")
    GT=[]
    for x in f:
        date_string = float(x.split('\n')[0])
        GT.append(date_string)
    return GT


def zoom(image):
    zoom = iaa.Affine(scale=(1, 1.3))
    image = zoom.augment_image(image)
    return image

def pan(image):
    pan = iaa.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)})
    image = pan.augment_image(image)
    return image

def img_random_flip(image):
    image = cv2.flip(image, 1)

    return image


def random_augment(image,steering):

    if np.random.rand() < 0.5:
        image = pan(image)
    if np.random.rand() < 0.5:
        image = zoom(image)
    if np.random.rand() < 0.5:
        image = img_random_flip(image)
        steering=-steering

    return image,steering

def img_preprocess(img):

    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img,  (3, 3), 0)
    img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)

    return img

def prepareData(labels_path, images_path, type=TYPE_FLOW_PRECOMPUTED):
    num_train_labels = 0
    train_labels = []
    train_images_pair_paths = []




    labels_string=load_labels(labels_path)
    num_bins = 10
    hist, bins = np.histogram(labels_string, num_bins)
    samples_per_bin = 4000


    filtered_list_indexes = []
    filtered_list_labels=[]
    for j in range(num_bins):
        list_ = []
        for i in range(len(labels_string)):
            if labels_string[i] >= bins[j] and labels_string[i] <= bins[j + 1]:
                list_.append(i)
        list_ = shuffle(list_)
        list_ = list_[0:samples_per_bin]

        for i in range(len(list_)):
            index=list_[i]
            label=labels_string[index]
            filtered_list_labels.append(label)
        filtered_list_indexes.extend(list_)

    print('balanced dataset:', len(filtered_list_indexes))


    hist, bins = np.histogram(filtered_list_labels, num_bins)


    for index in range(len(labels_string)):

        if index in filtered_list_indexes:
            print(index)
            train_labels.append(labels_string[index])
            if type == TYPE_FLOW_PRECOMPUTED:
                # Combine original and pre computed optical flow
                train_images_pair_paths.append( ( os.getcwd()+'/'+images_path[1:] + str(index)+ '.jpg',  os.getcwd() + images_path[1:] + str(index) + '.jpg',   os.getcwd() + images_path[1:] + str(index) + '.jpg',   os.getcwd() + images_path[1:] + str(index) + '.jpg',  os.getcwd() + images_path[1:] + str(index) + '.jpg') )
            else:
                # Combine 2 consecutive frames and calculate optical flow
                train_images_pair_paths.append( ( os.getcwd()+'/'+images_path[1:] + str(index-1)+ '.jpg',  os.getcwd() + images_path[1:] + str(index) + '.jpg') )

    return train_images_pair_paths, train_labels


def generatorData(samples, batch_size=32, type=TYPE_FLOW_PRECOMPUTED):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        samples = sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for imagePath, measurement in batch_samples:

                combined_image = None
                flow_image_bgr = None

                if type == 'augmentation':

                    # curr_image_path, flow_image_path = imagePath
                    # flow_image_bgr = cv2.imread(flow_image_path)
                    curr_image_path, flow_image_path1, flow_image_path2,flow_image_path3, flow_image_path4 = imagePath
                    #print(curr_image_path)

                    try:
                        im1=cv2.imread(curr_image_path)
                        im1 = cv2.resize(im1, (1920, 1080))

                    except:
                        print("An exception1 occurred :" ,flow_image_path1)


                    try:
                        flow_image_bgr,measurement=random_augment(im1,measurement)
                    except:
                        print("An exception occurred :", flow_image_path1)



                elif type == 'no-augmentation':
                    # curr_image_path, flow_image_path = imagePath
                    # flow_image_bgr = cv2.imread(flow_image_path)
                    curr_image_path, flow_image_path1, flow_image_path2,flow_image_path3, flow_image_path4 = imagePath

                    flow_image_bgr = cv2.imread(curr_image_path)
                    flow_image_bgr = cv2.resize(flow_image_bgr, (1920, 1080))


                combined_image=flow_image_bgr
                try:
                    combined_image = combined_image[1080 - 600:1080 - 200, 20:800]
                except:
                    print("An exception1 occurred :" ,flow_image_path1)

               # print(combined_image.shape)
                cv2.imwrite('preprosess.jpg', combined_image)
                combined_image = img_preprocess(combined_image)
                cv2.imwrite('processed.jpg', combined_image)
                combined_image = cv2.normalize(combined_image, None, alpha=-1, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)





                images.append(combined_image)
                angles.append(measurement)



            inputs = np.array(images)
            outputs = np.array(angles)
            yield sklearn.utils.shuffle(inputs, outputs)


if __name__ == '__main__':

    type_ = TYPE_FLOW_PRECOMPUTED   ## optical flow pre computed
    # type = TYPE_ORIGINAL

    train_images_pair_paths, train_labels =  prepareData(PATH_TRAIN_LABEL, PATH_TRAIN_IMAGES_FOLDER, type=type_)

    samples = list(zip(train_images_pair_paths, train_labels))
    train_samples, validation_samples = train_test_split(samples, test_size=0.2)

    print('Total Images: {}'.format( len(train_images_pair_paths)))
    print('Train samples: {}'.format(len(train_samples)))
    print('Validation samples: {}'.format(len(validation_samples)))

    training_generator = generatorData(train_samples, batch_size=BATCH_SIZE, type='augmentation')
    validation_generator = generatorData(validation_samples, batch_size=BATCH_SIZE, type='no-augmentation')

    print('Training model...')

    model = CNNModel()
    checkpoint_path='./'+MODEL_NAME+'.h5'
    model.load_weights(checkpoint_path)



    callbacks = [EarlyStopping(monitor='val_loss', patience=50),
             ModelCheckpoint(filepath=MODEL_NAME+'.h5', monitor='val_loss', save_best_only=True)]

    history_object = model.fit_generator(training_generator, samples_per_epoch= BATCH_SIZE, validation_data=validation_generator, \
                     validation_steps=50, callbacks=callbacks, epochs=EPOCH, verbose=1 )

    print('Training model complete...')

    print(history_object.history.keys())
    print('Loss')
    print(history_object.history['loss'])
    print('Validation Loss')
    print(history_object.history['val_loss'])


    plt.figure(figsize=[10,8])
    plt.plot(np.arange(1, len(history_object.history['loss'])+1), history_object.history['loss'],'r',linewidth=3.0)
    plt.plot(np.arange(1, len(history_object.history['val_loss'])+1), history_object.history['val_loss'],'b',linewidth=3.0)
    plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
    plt.xlabel('Epochs ',fontsize=16)
    plt.ylabel('Loss',fontsize=16)
    plt.title('Loss Curves',fontsize=16)
    plt.show()
    plt.savefig('graph.png')
