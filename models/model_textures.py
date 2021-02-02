import torch
import torch.nn as nn
import numpy as np
from skimage.io import imread

import neural_renderer as nr


class ModelTextures(nn.Module):
    def __init__(self, vertices, faces, filename_ref, camera_distance, camera_elevation, camera_azimuth):
        super(ModelTextures, self).__init__()

        self.register_buffer('vertices', vertices)
        self.register_buffer('faces', faces)

        # create textures
        texture_size = 2
        textures = torch.zeros(1, self.faces.shape[1], texture_size, texture_size, texture_size, 3, dtype=torch.float32)
        self.textures = nn.Parameter(textures)

        # load reference image
        image_ref = torch.from_numpy(imread(filename_ref).astype('float32') / 255.).permute(2,0,1)[None, ::]
        self.register_buffer('image_ref', image_ref)

        # save camera paraeters
        self.camera_distance = camera_distance
        self.camera_elevation = camera_elevation
        self.camera_azimuth = camera_azimuth

        # setup renderer
        renderer = nr.Renderer(camera_mode='look_at', far=600)
        # renderer.perspective = False
        renderer.light_intensity_directional = 0
        renderer.light_intensity_ambient = 1
        self.renderer = renderer

    def forward(self):
        # print(f"{self.camera_distance} - {self.camera_elevation} - {self.camera_azimuth}")
        self.renderer.eye = nr.get_points_from_angles(self.camera_distance, self.camera_elevation, self.camera_azimuth)
        image, _, _ = self.renderer(self.vertices, self.faces, torch.tanh(self.textures))
        loss = torch.sum((image - self.image_ref) ** 2)
        return loss