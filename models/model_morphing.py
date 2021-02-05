from __future__ import division
import os
import argparse
import glob

import torch
import torch.nn as nn
import numpy as np
from skimage.io import imread, imsave
import tqdm
import imageio

import neural_renderer as nr

current_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(current_dir, 'data')

class ModelMorphing(nn.Module):
    def __init__(self, vertices, faces, filename_ref, camera_distance, camera_elevation, camera_azimuth):
        super(ModelMorphing, self).__init__()

        self.vertices = nn.Parameter(vertices)
        self.register_buffer('faces', faces)

        # create textures
        texture_size = 2
        textures = torch.ones(1, self.faces.shape[1], texture_size, texture_size, texture_size, 3, dtype=torch.float32)
        self.register_buffer('textures', textures)

        # load reference image
        image_ref = torch.from_numpy(imread(filename_ref).astype(np.float32).mean(-1) / 255.)[None, ::]
        self.register_buffer('image_ref', image_ref)

        # save camera paraeters
        self.camera_distance = camera_distance
        self.camera_elevation = camera_elevation
        self.camera_azimuth = camera_azimuth

        # setup renderer
        renderer = nr.Renderer(camera_mode='look_at')
        self.renderer = renderer

    def forward(self):
        #self.renderer.eye = nr.get_points_from_angles(2.732, 0, 90)
        self.renderer.eye = nr.get_points_from_angles(self.camera_distance, self.camera_elevation, self.camera_azimuth)
        image = self.renderer(self.vertices, self.faces, mode='silhouettes')
        loss = torch.sum((image - self.image_ref[None, :, :])**2)
        return loss