import neural_renderer as nr
import torch
import torch.nn as nn
from skimage.io import imread


class ModelTextures(nn.Module):
    def __init__(self, vertices, faces, filename_ref, camera_distance, camera_elevation, camera_azimuth,
                 base_textures=None):
        super(ModelTextures, self).__init__()

        self.register_buffer('vertices', vertices)
        self.register_buffer('faces', faces)

        # Create textures
        # If no base textures are specified they are initialized to white
        if base_textures is None:
            texture_size = 2
            textures = torch.ones(1, self.faces.shape[1], texture_size, texture_size, texture_size, 3,
                                  dtype=torch.float32).cuda()
            self.textures = nn.Parameter(textures)
        else:
            self.textures = nn.Parameter(base_textures)

        # load reference image
        image_ref = torch.from_numpy(imread(filename_ref).astype('float32') / 255.).permute(2, 0, 1)[None, ::]
        image_ref = image_ref.cuda()
        self.register_buffer('image_ref', image_ref)

        # save camera parameters
        self.camera_x = camera_distance
        self.camera_y = camera_elevation
        self.camera_z = camera_azimuth

        # setup renderer
        renderer = nr.Renderer(camera_mode='look_at', far=200)
        renderer.light_intensity_directional = 0
        renderer.light_intensity_ambient = 1
        renderer.eye = (self.camera_x, self.camera_y, self.camera_z)
        self.renderer = renderer

    def forward(self):
        image, _, _ = self.renderer(self.vertices, self.faces, torch.tanh(self.textures))
        loss = torch.sum((image - self.image_ref) ** 2)
        return loss
