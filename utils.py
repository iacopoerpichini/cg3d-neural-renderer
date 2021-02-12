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
    if os.path.isdir(config.PATH.OUT):
        shutil.rmtree(config.PATH.OUT)
    os.mkdir(config.PATH.OUT)
    if os.path.isdir(config.PATH.TMP):
        shutil.rmtree(config.PATH.TMP)
    os.mkdir(config.PATH.TMP)


def make_gif(filename, dir_img):
    with imageio.get_writer(filename, mode='I') as writer:
        for filename in sorted(glob.glob(os.path.join(dir_img, '*.png'))):
            writer.append_data(imageio.imread(filename))
    writer.close()


def render_model(model, camera, config):
    # loop = tqdm(range(int(camera.azimuth - 120), int(camera.azimuth + 120), config.RENDERING_ANGLE_STEP))
    # loop = tqdm(range(int(camera.elevation - 120), int(camera.elevation + 120), config.RENDERING_ANGLE_STEP))
    loop = tqdm(range(int(0), int(180), config.RENDERING_ANGLE_STEP))

    dir_imgs = os.path.join(config.PATH.TMP, "render")
    if not os.path.isdir(dir_imgs):
        os.mkdir(dir_imgs)

    for num, theta in enumerate(loop):
        x = camera.z * math.cos(math.radians(theta))
        z = camera.z * math.sin(math.radians(theta))
        loop.set_description("Rendering final 3D model")
        # model.renderer.eye = nr.get_points_from_angles(camera.distance, elevation, camera.azimuth)
        # model.renderer.eye = get_points_from_angles(camera.distance, elevation, camera.azimuth)
        model.renderer.eye = (x, camera.y, z)
        images, _, _ = model.renderer(model.vertices, model.faces, torch.tanh(model.textures))
        image = images.detach().cpu().numpy()[0].transpose(1, 2, 0)
        imsave(os.path.join(dir_imgs, "%04d.png" % num), (255 * image).astype(np.uint8))

    make_gif(os.path.join(config.PATH.OUT, "rendered.gif"), dir_imgs)


def get_points_from_angles(distance, elevation, azimuth):
    elevation = math.radians(elevation)
    azimuth = math.radians(azimuth)

    x = distance*math.sin(elevation)*math.cos(azimuth)
    y = distance*math.sin(elevation)*math.sin(azimuth)
    z = distance*math.cos(elevation)

    return x, y, z


def get_angles_from_points(x, y, z):
    distance = math.sqrt(x ** 2 + y ** 2 + z ** 2)
    elevation = math.degrees(math.atan2(math.sqrt(x ** 2 + y ** 2), z))
    azimuth = math.degrees(math.atan2(y, x))

    return distance, elevation, azimuth

# def get_angles_from_points(x, z, y):
#     x1, y1, z1 = 0, 0, 0
#     x2, y2, z2 = x, y, z
#     distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
#     elevation = math.degrees(math.asin((z2 - z1) / distance))
#     azimuth = math.degrees(math.atan2((x2 - x1), (y2 - y1)))
#     # Manual fix: original code for function get_points_from_angles is wrong!
#     if x2 < 0 or y2 < 0:
#         distance = -distance
#     return distance, elevation, azimuth
