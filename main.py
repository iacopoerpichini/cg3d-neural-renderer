from __future__ import division
import os
import shutil
import argparse
import glob
from PIL import Image
from skimage.io import imsave
import torch
import tqdm
import imageio
import numpy as np
import neural_renderer as nr
from models.model_camera import ModelCamera
from models.model_textures import ModelTextures
import read_bfm

current_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(current_dir, 'data')
img_dir = os.path.join(data_dir, 'image_gif')
original_image_annotation = os.path.join(data_dir, 'original_image_annotation')
camera_opt_dir = os.path.join(data_dir, 'camera_opt_dir')

bfm = os.path.join(data_dir, 'model2017-1_bfm_nomouth.h5')
head = os.path.join(data_dir, 'head.obj')

# every time run create the folder
shutil.rmtree(img_dir)
os.mkdir(img_dir)
shutil.rmtree(img_dir)
os.mkdir(img_dir)

# other settings
camera_distance = -300
camera_elevation = -10
camera_azimuth = 0
texture_size = 2

iter_opt_camera = 5
iter_opt_textures = 2
use_bfm = True
swap_column = False


def make_gif(filename, dir_img):
    with imageio.get_writer(filename, mode='I') as writer:
        for filename in sorted(glob.glob(os.path.join(dir_img, '*.png'))):
            writer.append_data(imageio.imread(filename))
            # os.remove(filename)
    writer.close()


def optimize_model(model, iter_opt, model_type):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1, betas=(0.5, 0.999))
    loop = tqdm.tqdm(range(iter_opt))
    for i in loop:
        loop.set_description(f'Optimizing {model_type}')
        optimizer.zero_grad()
        loss = model()
        loss.backward()
        optimizer.step()
        if model_type == 'camera':
            images, _, _ = model.renderer(model.vertices, model.faces, torch.tanh(model.textures))
            image = images.detach().cpu().numpy()[0].transpose(1, 2, 0)
            im = Image.fromarray((255 * image).astype(np.uint8))
            im.save(os.path.join(camera_opt_dir, '%04d.png' % i))
            # loop.set_description('Optimizing (loss %.4f)' % loss.data)

    if model_type == "camera":
        make_gif(os.path.join(data_dir, 'camera.gif'), camera_opt_dir)
        """save silouette debugging"""
        image = model.renderer(model.vertices, model.faces, mode='silhouettes')
        image = image.detach().cpu().numpy().transpose(1, 2, 0)
        imsave(os.path.join(data_dir, "franco.png"), (255 * image).astype(np.uint8))



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-it', '--filename_textures', type=str, default=os.path.join(data_dir, 'resize.png'))
    parser.add_argument('-is', '--filename_silouette', type=str, default=os.path.join(original_image_annotation, '00039_skin_resize.png'))
    parser.add_argument('-or', '--filename_output', type=str, default=os.path.join(data_dir, 'result.gif'))
    parser.add_argument('-g', '--gpu', type=int, default=0)
    args = parser.parse_args()

    if use_bfm:
        vertices, faces = read_bfm.read_vertices_and_faces_from_file(bfm, swap_column=swap_column)
    else:
        vertices, faces = read_bfm.read_obj(head)

    # create texture [batch_size=1, num_faces, texture_size, texture_size, texture_size, RGB]
    # textures = torch.ones(1, model.faces.shape[1], texture_size, texture_size, texture_size, 3, dtype=torch.float32).cuda()

    # optimize camera position
    model = ModelCamera(vertices, faces, args.filename_silouette, camera_distance, camera_elevation, camera_azimuth)
    model.cuda()
    optimize_model(model, iter_opt_camera, model_type='camera')

    # getting camera position optimized parameters
    camera_position = model.camera_position.cpu().detach().numpy()
    # Convert numpy scalar to python types to avoid errors in neural renderer functions
    camera_distance_start, camera_elevation_start, camera_azimuth_start = float(camera_position[0]), float(camera_position[1]), float(camera_position[2])
    print("------- OPTIMIZED CAMERA POS. --------")
    print(f"Distance: {camera_distance_start}")
    print(f"Elevation: {camera_elevation_start}")
    print(f"Azimuth: {camera_azimuth_start}")

    # optimize textures to apply the face image to the model
    model = ModelTextures(model.vertices, model.faces, args.filename_textures, camera_distance_start, camera_elevation_start, camera_azimuth_start)
    model.cuda()
    optimize_model(model, iter_opt_textures, model_type='textures')

    """save silouette debugging"""
    image = model.renderer(model.vertices, model.faces, mode='silhouettes')
    image = image.detach().cpu().numpy().transpose(1, 2, 0)
    imsave(os.path.join(data_dir, "pippo.png"), (255 * image).astype(np.uint8))

    # draw object
    loop = tqdm.tqdm(range(-120, 120, 4))
    for num, azimuth in enumerate(loop):
        loop.set_description('Drawing')
        model.renderer.eye = nr.get_points_from_angles(camera_distance, camera_elevation, azimuth)
        images, _, _ = model.renderer(model.vertices, model.faces, torch.tanh(model.textures))
        image = images.detach().cpu().numpy()[0].transpose((1, 2, 0))
        im = Image.fromarray((255 * image).astype(np.uint8))
        im.save(os.path.join(img_dir, '%04d.png' % num))

    make_gif(args.filename_output, img_dir)


if __name__ == '__main__':
    main()