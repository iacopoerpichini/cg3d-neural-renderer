import math

import neural_renderer as nr
import numpy as np
import torch
import torch.nn as nn
from skimage.io import imread

from mesh.read_bfm_2009 import filter_region, RegionType


class ModelCamera(nn.Module):
    def __init__(self, vertices, faces, regions, silhouette_face, start_x, start_y, start_z, max_rotation_l_r=10,
                 max_rotation_u_d=10, weight_loss_constr=2e4, use_anchor_points=False, silhouette_nose=None,
                 silhouette_mouth=None, weight_loss_anchor=1):
        super(ModelCamera, self).__init__()

        self.register_buffer('vertices', torch.tensor(vertices, requires_grad=True))
        self.register_buffer('faces', faces)

        # create textures
        texture_size = 2
        textures = torch.ones(1, self.faces.shape[1], texture_size, texture_size, texture_size, 3, dtype=torch.float32)
        self.register_buffer('textures', textures)

        # load reference image
        img_silhouette_face = torch.from_numpy((imread(silhouette_face).max(-1) != 0).astype(np.float32))
        self.register_buffer('img_silhouette_face', img_silhouette_face)

        # camera parameters
        self.start_x = start_x
        self.start_y = start_y
        self.start_z = start_z
        # self.x_param = nn.Parameter(torch.tensor(np.random.rand(1)*0.1, dtype=torch.float32).cuda())
        # self.y_param = nn.Parameter(torch.tensor(np.random.rand(1)*0.1, dtype=torch.float32).cuda())
        self.camera_x = nn.Parameter(torch.tensor([start_x], dtype=torch.float32).cuda())
        self.camera_y = nn.Parameter(torch.tensor([start_y], dtype=torch.float32).cuda())
        self.camera_z = nn.Parameter(torch.tensor([start_z], dtype=torch.float32).cuda())

        # Setup camera constraints parameters
        self.epsilon = 0.01
        self.weight_loss_constr = weight_loss_constr
        self.register_buffer('max_rotation_l_r', torch.tensor([max_rotation_l_r], dtype=torch.float32))
        self.register_buffer('max_rotation_u_d', torch.tensor([max_rotation_u_d], dtype=torch.float32))
        self.register_buffer('start_angle_l_r', torch.tensor([math.degrees(math.atan2(start_x, start_z))], dtype=torch.float32))
        self.register_buffer('start_angle_u_d', torch.tensor([math.degrees(math.atan2(start_y, start_z))], dtype=torch.float32))

        # Setup anchor points parameters
        self.use_anchor_points = use_anchor_points
        self.weight_loss_anchor = weight_loss_anchor
        if use_anchor_points:
            print('Splitting nose')
            _, self.triangles_nose = filter_region(vertices, faces, regions, RegionType.NOSE)
            print('Splitting mouth')
            _, self.triangles_mouth = filter_region(vertices, faces, regions, RegionType.MOUTH)
            silhouette_nose = torch.from_numpy((imread(silhouette_nose).max(-1) != 0).astype(np.float32))
            self.register_buffer('img_silhouette_nose', silhouette_nose)
            silhouette_mouth = torch.from_numpy((imread(silhouette_mouth).max(-1) != 0).astype(np.float32))
            self.register_buffer('img_silhouette_mouth', silhouette_mouth)

        # setup renderer
        renderer = nr.Renderer(camera_mode='look_at', far=200)
        # renderer.eye = self.camera_position
        self.renderer = renderer

    def forward(self):
        # x_param, y_param, camera_z = self.camera_params
        # angle_l_r = self.start_angle_l_r + self.max_rotation_l_r * torch.tanh(self.x_param)
        # angle_u_d = self.start_angle_u_d + self.max_rotation_u_d * torch.tanh(self.y_param)
        angle_l_r = self.start_angle_l_r + self.max_rotation_l_r * self.camera_x
        angle_u_d = self.start_angle_u_d + self.max_rotation_u_d * self.camera_y
        # camera_x = torch.sin(angle_l_r*math.pi/180.)
        # camera_y = torch.sin(angle_u_d*math.pi/180.)

        self.renderer.eye = torch.cat([self.camera_x, self.camera_y, self.camera_z])
        image = self.renderer(self.vertices, self.faces, mode='silhouettes')
        loss_image = torch.sum((image - self.img_silhouette_face[None, :, :]) ** 2)

        loss_constraints = self._get_position_constraints_loss(angle_l_r, angle_u_d)

        loss_anchor_points = self._get_anchor_points_loss()

        return loss_image + loss_constraints + self.weight_loss_anchor * loss_anchor_points

    def _get_position_constraints_loss(self, angle_l_r, angle_u_d):
        loss_rotation_l_r = torch.exp(self.weight_loss_constr * 1. / ((self.max_rotation_l_r ** 2 - torch.min(self.max_rotation_l_r - self.epsilon,
                                                                      self.start_angle_l_r - angle_l_r) ** 2) ** 2))
        loss_rotation_u_d = torch.exp(self.weight_loss_constr * 1. / ((self.max_rotation_u_d ** 2 - torch.min(self.max_rotation_u_d - self.epsilon,
                                                                      self.start_angle_u_d - angle_u_d) ** 2) ** 2))

        return loss_rotation_l_r + loss_rotation_u_d

    def _get_anchor_points_loss(self):
        if self.use_anchor_points:
            img_nose = self.renderer(self.vertices, self.triangles_nose, mode='silhouettes')
            img_mouth = self.renderer(self.vertices, self.triangles_mouth, mode='silhouettes')
            loss_nose = torch.sum((img_nose - self.img_silhouette_nose[None, :, :]) ** 2)
            loss_mouth = torch.sum((img_mouth - self.img_silhouette_mouth[None, :, :]) ** 2)
            return loss_nose + loss_mouth
        else:
            return 0
