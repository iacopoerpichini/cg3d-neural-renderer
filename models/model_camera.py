import numpy as np
import torch
import torch.nn as nn
from skimage.io import imread
import neural_renderer as nr
import math
from torch.autograd import Variable

def get_angles_from_points(x, z, y):
    x1, y1, z1 = 0, 0, 0
    x2, y2, z2 =  x, y, z
    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
    elevation = math.degrees(math.asin((z2 - z1) / distance))
    azimuth = math.degrees(math.atan2((x2 - x1), (y2 - y1)))
    return (distance,elevation,azimuth)


class ModelCamera(nn.Module):
    def __init__(self, vertices, faces, ref_silhouette, start_distance, start_elevation, start_azimuth, max_elevation=5, max_azimuth=5, epsilon=0.01):
        super(ModelCamera, self).__init__()

        self.register_buffer('vertices', vertices)
        self.register_buffer('faces', faces)

        # create textures
        texture_size = 2
        textures = torch.ones(1, self.faces.shape[1], texture_size, texture_size, texture_size, 3, dtype=torch.float32)
        self.register_buffer('textures', textures)

        # load reference image
        #image_ref = torch.from_numpy(imread(ref_silhouette).astype('float32') / 255.).permute(2, 0, 1)[None, ::]
        image_ref = torch.from_numpy((imread(ref_silhouette).max(-1) != 0).astype(np.float32))
        self.register_buffer('image_ref', image_ref)

        # camera parameters
        self.start_distance = start_distance
        self.start_elevation = start_elevation
        self.start_azimuth = start_azimuth % 180
        self.camera_position = nn.Parameter(torch.from_numpy(np.array(nr.get_points_from_angles(float(start_distance), float(start_elevation), float(start_azimuth)), dtype=np.float32)))

        self.max_elevation = torch.tensor(max_elevation).cuda()
        self.max_azimuth = torch.tensor(max_azimuth).cuda()
        self.epsilon = epsilon

        # setup renderer
        renderer = nr.Renderer(camera_mode='look_at', far=600)
        renderer.eye = self.camera_position
        # renderer.perspective = False
        # renderer.light_intensity_directional = 0
        # renderer.light_intensity_ambient = 1
        self.renderer = renderer

    def forward(self):
        curr_distance = self.camera_position[0]
        curr_elevation = self.camera_position[1]
        curr_azimuth = self.camera_position[2] % 180

        curr_distance, curr_elevation, curr_azimuth = get_angles_from_points(curr_distance, curr_elevation, curr_azimuth)


        image = self.renderer(self.vertices, self.faces, mode='silhouettes')



        # loss_elevation = 1/torch.pow(torch.pow(self.max_elevation, 2) - torch.pow(torch.abs(torch.Tensor([curr_elevation - self.start_elevation]).cuda()), 2), 2)
        # loss_azimuth = 1/torch.pow(torch.pow(self.max_azimuth, 2) - torch.pow(torch.abs(torch.Tensor([curr_azimuth - self.start_azimuth]).cuda()), 2), 2)

        # loss_elevation = 1./((float(self.max_elevation) - (self.start_elevation - curr_elevation))**2)
        loss_elevation = 1./((float(self.max_elevation) - min(float(self.max_elevation) - self.epsilon, self.start_elevation - curr_elevation))**2)
        # loss_azimuth = 1./((float(self.max_azimuth) - (self.start_azimuth - curr_azimuth))**2)
        loss_azimuth = 1./((float(self.max_azimuth) - min(float(self.max_azimuth) - self.epsilon, self.start_azimuth - curr_azimuth))**2)

        loss_image = torch.sum((image - self.image_ref[None, :, :]) ** 2)
        #loss_image = 0

        loss_elevation = loss_elevation*1e7
        loss_azimuth = loss_azimuth*1e7

        # print(f"\nstart parameters: {self.start_distance} - {self.start_elevation} - {self.start_azimuth}")
        # print(f"curr. parameters: {curr_distance} - {curr_elevation} - {curr_azimuth}")
        # print(f"max parameters: {self.max_elevation} - {self.max_azimuth}")
        # print(f"\ndenom: {1/((float(self.max_azimuth) - (self.start_azimuth - curr_azimuth))**2)}")
        # print(f"loss el: {loss_elevation}")
        # print(f"loss az: {loss_azimuth}")
        # print(f"loss im: {loss_image}")
        # print(loss_image + loss_elevation + loss_azimuth)

        return loss_image #+ loss_elevation + loss_azimuth