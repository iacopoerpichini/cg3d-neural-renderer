from __future__ import division

import argparse

import torch

from camera import Camera
from config import get_config_defaults, merge_with_cmd_params
from mesh import read_bfm_mesh, Mesh
from optimization import get_optimized_model_camera, get_optimized_model_morphing, get_optimized_model_textures
from utils import render_model, clean_output_dirs

torch.cuda.set_device(1)


def read_cmd_params():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--example_input", type=str, default="ex-1")

    return parser.parse_args()


def main():
    args = read_cmd_params()

    # Get configuration
    config = get_config_defaults()
    merge_with_cmd_params(config, args)

    clean_output_dirs(config)

    # Load the 3D mesh model
    print("Loading BFM model...")
    mesh = read_bfm_mesh(config)

    # Optimize the camera position using the reference silhouette
    print("\n------- STARTING CAMERA OPTIMIZATION -------")
    camera = Camera(config.CAMERA.START_X, config.CAMERA.START_Y, config.CAMERA.START_Z)
    model = get_optimized_model_camera(mesh, camera, config)
    print("------- CAMERA OPTIMIZATION ENDED -------")

    # Getting camera position optimized parameters
    camera = Camera(*model.renderer.eye)

    # Morph the model to fit the reference silhouette
    print("\n------- STARTING MODEL MORPHING -------")
    if config.MORPHING:
        model = get_optimized_model_morphing(mesh, camera, config)
    print("------- MODEL MORPHING ENDED -------")

    # Optimize model textures to apply the face image to it
    print("\n------- STARTING TEXTURES OPTIMIZATION -------")
    mesh = Mesh(model.vertices, model.faces, mesh.textures)
    model = get_optimized_model_textures(mesh, camera, config)
    print("------- TEXTURES OPTIMIZATION ENDED -------")

    # Draw the final optimized mesh
    print("\nFinalizing...")
    render_model(model, camera, config)


if __name__ == '__main__':
    main()
