import numpy as np
import torch
import torch.nn as nn
from skimage.io import imread
import neural_renderer as nr
import math
from torch.autograd import Variable

from read_bfm_2009 import filter_region, RegionType


def get_angles_from_points(x, z, y):
    x1, y1, z1 = 0, 0, 0
    x2, y2, z2 =  x, y, z
    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
    elevation = math.degrees(math.asin((z2 - z1) / distance))
    azimuth = math.degrees(math.atan2((x2 - x1), (y2 - y1)))
    return (distance,elevation,azimuth)


class ModelCamera(nn.Module):
    def __init__(self, vertices, faces, regions, silhouette_face, start_distance, start_elevation, start_azimuth, max_elevation=10,
                 max_azimuth=10, epsilon=0.01, use_regions=False, silhouette_nose=None, silhouette_mouth=None):
        super(ModelCamera, self).__init__()

        self.use_regions = use_regions
        if use_regions:
            print('Splitting nose')
            _, self.triangles_nose = filter_region(vertices, faces, regions, RegionType.NOSE)
            print('Splitting mouth')
            _, self.triangles_mouth = filter_region(vertices, faces, regions, RegionType.MOUTH)
            silhouette_nose = torch.from_numpy((imread(silhouette_nose).max(-1) != 0).astype(np.float32))
            self.register_buffer('img_silhouette_nose', silhouette_nose)
            silhouette_mouth = torch.from_numpy((imread(silhouette_mouth).max(-1) != 0).astype(np.float32))
            self.register_buffer('img_silhouette_mouth', silhouette_mouth)


        self.register_buffer('vertices', vertices)
        self.register_buffer('faces', faces)

        # create textures
        texture_size = 2
        textures = torch.ones(1, self.faces.shape[1], texture_size, texture_size, texture_size, 3, dtype=torch.float32)
        self.register_buffer('textures', textures)

        # load reference image
        #image_ref = torch.from_numpy(imread(ref_silhouette).astype('float32') / 255.).permute(2, 0, 1)[None, ::]
        img_silhouette_face = torch.from_numpy((imread(silhouette_face).max(-1) != 0).astype(np.float32))
        self.register_buffer('img_silhouette_face', img_silhouette_face)

        # camera parameters
        self.start_distance = start_distance
        self.start_elevation = start_elevation
        self.start_azimuth = start_azimuth % 180

        #torch.from_numpy(np.array(nr.get_points_from_angles(float(start_distance), float(start_elevation), float(start_azimuth)), dtype=np.float32))
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
        curr_azimuth = self.camera_position[2]

        curr_distance, curr_elevation, curr_azimuth = get_angles_from_points(curr_distance, curr_elevation, curr_azimuth)

        image = self.renderer(self.vertices, self.faces, mode='silhouettes')

        # loss_elevation = 1./((float(self.max_elevation) - (self.start_elevation - curr_elevation))**2)
        loss_elevation = 1./((float(self.max_elevation) - min(float(self.max_elevation) - self.epsilon, self.start_elevation - curr_elevation))**2)
        # loss_azimuth = 1./((float(self.max_azimuth) - (self.start_azimuth - curr_azimuth))**2)
        loss_azimuth = 1./((float(self.max_azimuth)**2 - min(float(self.max_azimuth) - self.epsilon, self.start_azimuth - curr_azimuth)**2)**2)

        loss_elevation = loss_elevation*1e6
        loss_azimuth = loss_azimuth*1e6

        loss_image = torch.sum((image - self.img_silhouette_face[None, :, :]) ** 2)

        loss_regions = 0
        if self.use_regions:
            img_nose = self.renderer(self.vertices, self.triangles_nose, mode='silhouettes')
            img_mouth = self.renderer(self.vertices, self.triangles_mouth, mode='silhouettes')
            loss_nose = torch.sum((img_nose - self.img_silhouette_nose[None, :, :]) ** 2)
            loss_mouth = torch.sum((img_mouth - self.img_silhouette_mouth[None, :, :]) ** 2)
            loss_regions = loss_nose + loss_mouth

        # print(f"Loss image: {loss_image}")
        # print(f"Loss regions: {loss_regions}")

        return loss_image + loss_elevation + loss_azimuth + loss_regions*10