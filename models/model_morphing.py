from __future__ import division

import os

import neural_renderer as nr
import numpy as np
import torch
import torch.nn as nn
from skimage.io import imread

current_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(current_dir, 'data')

class ModelMorphing(nn.Module):
    def __init__(self, vertices, faces, filename_ref, camera_x, camera_y, camera_z):
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

        # save camera parameters
        self.camera_x = camera_x
        self.camera_y = camera_y
        self.camera_z = camera_z

        # setup renderer
        renderer = nr.Renderer(camera_mode='look_at', far=200)
        renderer.eye = (self.camera_x, self.camera_y, self.camera_z)
        self.renderer = renderer

    def forward(self):
        image = self.renderer(self.vertices, self.faces, mode='silhouettes')
        loss = torch.sum((image - self.image_ref[None, :, :])**2)
        return loss