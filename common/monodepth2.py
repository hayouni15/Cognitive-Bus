# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import sys
import glob
import argparse
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm

import torch
from torchvision import transforms, datasets

from ..monodepth2.networks.resnet_encoder import ResnetEncoder
from ..monodepth2.networks.depth_decoder import DepthDecoder
from ..monodepth2.networks import pose_cnn
from ..monodepth2.networks import pose_decoder
from ..monodepth2.layers import disp_to_depth
from ..monodepth2.utils import download_model_if_doesnt_exist

class Depth:
    def __init__(self, model_name):
        self.model_name = model_name
        #self.checkpoint_dir = checkpoint_dir

    def start_depth(self):
        self.device = torch.device("cuda")
        model_path='/home/ahayouni/Documents/brite-unit2/src/monodepth2/models/'+str(self.model_name)
        print("-> Loading model from ", model_path)
        encoder_path = os.path.join(model_path, "encoder.pth")
        self.depth_decoder_path = os.path.join(model_path, "depth.pth")
        # LOADING PRETRAINED MODEL
        print("   Loading pretrained encoder")
        self.encoder = ResnetEncoder(18, False)
        self.loaded_dict_enc = torch.load(encoder_path, map_location=self.device)

        # extract the height and width of image that this model was trained with
        self.feed_height = self.loaded_dict_enc['height']
        self.feed_width = self.loaded_dict_enc['width']
        filtered_dict_enc = {k: v for k, v in self.loaded_dict_enc.items() if k in self.encoder.state_dict()}
        self.encoder.load_state_dict(filtered_dict_enc)
        self.encoder.to(self.device)
        self.encoder.eval()

        print("   Loading pretrained decoder")
        self.depth_decoder = DepthDecoder(
            num_ch_enc=self.encoder.num_ch_enc, scales=range(4))

        loaded_dict = torch.load(self.depth_decoder_path, map_location=self.device)
        self.depth_decoder.load_state_dict(loaded_dict)

        self.depth_decoder.to(self.device)
        self.depth_decoder.eval()

    def get_depth(self,img):
        """Function to predict for a single image or folder of images
        """
        # PREDICTING ON EACH IMAGE IN TURN
        with torch.no_grad():

            # Load image and preprocess
            input_image = pil.fromarray(img)
            original_width, original_height= input_image.size
            input_image = input_image.resize((self.feed_width, self.feed_height), pil.LANCZOS)
            input_image = transforms.ToTensor()(input_image).unsqueeze(0)

            # PREDICTION
            input_image = input_image.to(self.device)
            features = self.encoder(input_image)
            outputs = self.depth_decoder(features)

            disp = outputs[("disp", 0)]
            disp_resized = torch.nn.functional.interpolate(
                disp, (original_height, original_width), mode="bilinear", align_corners=False)

            # Saving numpy file
            #output_name = os.path.splitext(os.path.basename(image_path))[0]
            #name_dest_npy = os.path.join(output_directory, "{}_disp.npy".format(output_name))
            scaled_disp, _ = disp_to_depth(disp, 0.1, 100) # prediction
            #np.save(name_dest_npy, scaled_disp.cpu().numpy())

            # Saving colormapped depth image
            disp_resized_np = disp_resized.squeeze().cpu().numpy()
            vmax = np.percentile(disp_resized_np, 95)
            normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
            mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
            colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
            im = pil.fromarray(colormapped_im)

            #name_dest_im = os.path.join(output_directory, "{}_disp.jpeg".format(output_name))
            #im.save(name_dest_im)

            return np.asarray(im)

