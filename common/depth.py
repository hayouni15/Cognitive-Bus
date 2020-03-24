import tensorflow as tf
import sys
import os
import argparse
import time
import datetime
from ..monodepth.utils import *
from ..monodepth.pydnet import *


class Depth:
    def __init__(self, resolution, checkpoint_dir):
        self.resolution = resolution
        self.checkpoint_dir = checkpoint_dir

    def load(self):

        self.placeholders = {'im0': tf.placeholder(tf.float32, [None, None, None, 3], name='im0')}

        with tf.variable_scope("model") as scope:
            self.model = pydnet(self.placeholders)

        self.init = tf.group(tf.global_variables_initializer(),
                        tf.local_variables_initializer())
        self.loader = tf.train.Saver()
        self.saver = tf.train.Saver()

    def estimate_depth(self,img):
        height = 256*3
        width = 512*3
        sess=self.sess

        sess.run(self.init)
        self.loader.restore(sess, self.checkpoint_dir)

        img = cv2.resize(img, (width, height)).astype(np.float32) / 255
        img = np.expand_dims(img, 0)
        start = time.time()
        disp = sess.run(self.model.results[self.resolution - 1], feed_dict={self.placeholders['im0']: img})
        end = time.time()

        disp_color = applyColorMap(disp[0, :, :, 0] * 20, 'plasma')

        return  disp_color



    def depth_map(self,img):
        self.img=img
        tf.app.run(self.estimate_depth,[self])

