import math
import shutil

import torch
import imageio
import glob
import os
import neural_renderer as nr
import numpy as np
from skimage.io import imsave
from tqdm import tqdm


def clean_output_dirs(config):
    shutil.rmtree(config.PATH.OUT)
    os.mkdir(config.PATH.OUT)
    shutil.rmtree(config.PATH.TMP)
    os.mkdir(config.PATH.TMP)


def make_gif(filename, dir_img):
    with imageio.get_writer(filename, mode='I') as writer:
        for filename in sorted(glob.glob(os.path.join(dir_img, '*.png'))):
            writer.append_data(imageio.imread(filename))
    writer.close()


def render_model(model, camera, config):
    loop = tqdm(range(int(camera.azimuth - 120), int(camera.azimuth + 120), config.RENDERING_ANGLE_STEP))

    dir_imgs = os.path.join(config.PATH.TMP, "render")
    if not os.path.isdir(dir_imgs):
        os.mkdir(dir_imgs)

    for num, azimuth in enumerate(loop):
        loop.set_description("Rendering final 3D model")
        model.renderer.eye = nr.get_points_from_angles(camera.distance, camera.elevation, azimuth)
        images, _, _ = model.renderer(model.vertices, model.faces, torch.tanh(model.textures))
        image = images.detach().cpu().numpy()[0].transpose(1, 2, 0)
        imsave(os.path.join(dir_imgs, "%04d.png" % num), (255 * image).astype(np.uint8))

    make_gif(os.path.join(config.PATH.OUT, "rendered.gif"), dir_imgs)


def get_angles_from_points(x, z, y):
    x1, y1, z1 = 0, 0, 0
    x2, y2, z2 = x, y, z
    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
    elevation = math.degrees(math.asin((z2 - z1) / distance))
    azimuth = math.degrees(math.atan2((x2 - x1), (y2 - y1)))
    if x2 < 0:
        return -distance, -elevation, azimuth
    return distance, elevation, azimuth
