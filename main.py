from __future__ import division

import torch

from camera import Camera
from config import get_config_defaults
from mesh import read_bfm_mesh, Mesh
from optimization import get_optimized_model_camera, get_optimized_model_morphing, get_optimized_model_textures
from utils import render_model, clean_output_dirs

torch.cuda.set_device(1)


def main():
    # Get configuration
    config = get_config_defaults()

    clean_output_dirs(config)

    # Load the 3D mesh model
    mesh = read_bfm_mesh(config)

    # Optimize the camera position using the reference silhouette
    camera = Camera(config.CAMERA.START_X, config.CAMERA.START_Y, config.CAMERA.START_Z)
    model = get_optimized_model_camera(mesh, camera, config)

    # Getting camera position optimized parameters
    camera = Camera(*model.renderer.eye)

    # Morph the model to fit the reference silhouette
    if config.MORPHING:
        model = get_optimized_model_morphing(mesh, camera, config)
    # Optimize model textures to apply the face image to it
    mesh = Mesh(model.vertices, model.faces)
    model = get_optimized_model_textures(mesh, camera, config)

    # Draw the final optimized mesh
    render_model(model, camera, config)


if __name__ == '__main__':
    main()
