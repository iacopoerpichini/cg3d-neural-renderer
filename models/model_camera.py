import numpy as np
import torch
import torch.nn as nn
from skimage.io import imread
import neural_renderer as nr


class ModelCamera(nn.Module):
    def __init__(self, vertices, faces, ref_silhouette, start_distance, start_elevation, start_azimuth, max_elevation=30, max_azimuth=30):
        super(ModelCamera, self).__init__()

        self.register_buffer('vertices', vertices)
        self.register_buffer('faces', faces)

        # create textures
        texture_size = 2
        textures = torch.ones(1, self.faces.shape[1], texture_size, texture_size, texture_size, 3, dtype=torch.float32)
        self.register_buffer('textures', textures)

        # load reference image
        image_ref = torch.from_numpy(imread(ref_silhouette).astype('float32') / 255.).permute(2, 0, 1)[None, ::]
        self.register_buffer('image_ref', image_ref)

        # camera parameters
        self.camera_position = nn.Parameter(torch.from_numpy(np.array([start_distance, start_elevation, start_azimuth], dtype=np.float32)))
        #self.camera_position = nn.Parameter(torch.from_numpy(np.array(nr.get_points_from_angles(float(start_distance), float(start_elevation), float(start_azimuth)), dtype=np.float32)))

        self.max_elevation = torch.tensor(max_elevation).cuda()
        self.max_azimuth = torch.tensor(max_azimuth).cuda()

        # setup renderer
        renderer = nr.Renderer(camera_mode='look_at', far=600)
        renderer.eye = self.camera_position
        # renderer.light_intensity_directional = 0
        # renderer.light_intensity_ambient = 1
        self.renderer = renderer

    def forward(self):
        # curr_distance = self.camera_position[0]
        # curr_elevation = self.camera_position[1]
        # curr_azimuth = self.camera_position[2]

        image = self.renderer(self.vertices, self.faces, mode='silhouettes')

        # loss_elevation = 1/torch.pow(torch.pow(self.max_elevation, 2) - torch.pow(torch.abs(curr_elevation), 2), 2)
        # loss_azimuth = 1/torch.pow(torch.pow(self.max_azimuth, 2) - torch.pow(torch.abs(curr_azimuth), 2), 2)
        # loss_elevation = 0
        # loss_azimuth = 0
        loss_image = torch.sum((image - self.image_ref[None, :, :]) ** 2)

        return loss_image# + loss_elevation + loss_azimuth
