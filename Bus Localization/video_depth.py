#
# MIT License
#
# Copyright (c) 2018 Matteo Poggi m.poggi@unibo.it
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import tensorflow as tf
import sys
import os
import argparse
import time
import datetime
from ..monodepth.utils import *
from ..monodepth.pydnet import *

# forces tensorflow to run on CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

parser = argparse.ArgumentParser(description='Argument parser')

""" Arguments related to network architecture"""
parser.add_argument('--width', dest='width', type=int, default=512, help='width of input images')
parser.add_argument('--height', dest='height', type=int, default=256, help='height of input images')
parser.add_argument('--resolution', dest='resolution', type=int, default=1, help='resolution [1:H, 2:Q, 3:E]')
parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', type=str, default='checkpoint/IROS18/pydnet', help='checkpoint directory')

args = parser.parse_args()

def main():
  checkpoint_dir='/home/ahayouni/Documents/brite-unit2/src/monodepth/pydnet'
  resolution=1

  height = args.height
  width = args.width
  placeholders = {'im0':tf.placeholder(tf.float32,[None, None, None, 3], name='im0')}

  with tf.variable_scope("model") as scope:
    model = pydnet(placeholders)

  init = tf.group(tf.global_variables_initializer(),
                  tf.local_variables_initializer())

  loader = tf.train.Saver()
  saver = tf.train.Saver()
  Front_right_cam_file = '/media/ahayouni/Elements/simulation_data/2019-08-05 13-44-18 Avant droit.mp4'
  #cam = cv2.VideoCapture(0)
  cam = cv2.VideoCapture(Front_right_cam_file)
  with tf.Session() as sess:
    sess.run(init)
    loader.restore(sess, checkpoint_dir)

    while True:

      ret_val, img = cam.read()
      img = cv2.resize(img, (width, height)).astype(np.float32) / 255
      img = np.expand_dims(img, 0)
      start = time.time()
      disp = sess.run(model.results[resolution-1], feed_dict={placeholders['im0']: img})
      end = time.time()

      disp_color = applyColorMap(disp[0,:,:,0]*20, 'plasma')


      cv2.imshow('pydnet', img[0])
      cv2.imshow('disp_color', disp_color)
      k = cv2.waitKey(1)


      print("Time: " + str(end - start))


    cam.release()

if __name__ == '__main__':
  tf.app.run(main())
