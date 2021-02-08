from __future__ import division

from config import get_config_defaults
from camera import Camera
from mesh import read_bfm_mesh, Mesh
from optimization import get_optimized_model_camera, get_optimized_model_morphing, get_optimized_model_textures
from utils import render_model, get_angles_from_points, clean_output_dirs


# torch.cuda.set_device(1)


def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-it', '--filename_textures', type=str, default=os.path.join(data_dir,'model_b', '13_resize.png'))
    # parser.add_argument('-is', '--filename_silouette', type=str, default=os.path.join(data_dir,'model_b', '00013_skin_resize.png'))#'silouette.png'))
    # parser.add_argument('-or', '--filename_output', type=str, default=os.path.join(data_dir, 'result.gif'))
    # parser.add_argument('-g', '--gpu', type=int, default=0)
    # args = parser.parse_args()

    config = get_config_defaults()

    clean_output_dirs(config)

    # Load the 3D mesh model
    mesh = read_bfm_mesh(config)

    # Optimize the camera position using the reference silhouette
    camera = Camera(config.CAMERA.START_DISTANCE, config.CAMERA.START_ELEVATION, config.CAMERA.START_AZIMUTH)
    model = get_optimized_model_camera(mesh, camera, config)

    # Getting camera position optimized parameters
    # (Convert numpy scalar to python types to avoid errors in neural renderer functions)
    camera_position = model.camera_position.cpu().detach().numpy()
    camera = Camera(*get_angles_from_points(float(camera_position[0]), float(camera_position[1]), float(camera_position[2])))

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
