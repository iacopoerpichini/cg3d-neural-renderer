import os
import torch
from skimage.io import imsave
from tqdm import tqdm
import numpy as np

from models.model_camera import ModelCamera
from models.model_morphing import ModelMorphing
from models.model_textures import ModelTextures
from utils import make_gif


def get_optimized_model_camera(mesh, camera, config):
    model = ModelCamera(mesh.vertices, mesh.faces, mesh.regions, config.CAMERA.REF_SILHOUETTE,
                        camera.x, camera.y, camera.z, min_distance=config.CAMERA.MIN_DISTANCE,
                        max_rotation_l_r=config.CAMERA.MAX_ROTATION_LR, max_rotation_u_d=config.CAMERA.MAX_ROTATION_UD,
                        use_anchor_points=config.CAMERA.USE_ANCHOR_POINTS,
                        silhouette_nose=config.CAMERA.ANCHOR_NOSE_IMG, silhouette_mouth=config.CAMERA.ANCHOR_MOUTH_IMG)
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.OPT.LR_CAMERA)

    dir_imgs = os.path.join(config.PATH.TMP, "camera")
    if not os.path.isdir(dir_imgs):
        os.mkdir(dir_imgs)

    loop = tqdm(range(config.OPT.ITER_CAMERA))
    for i in loop:
        loop.set_description(f"Optimizing camera")
        optimizer.zero_grad()
        loss = model()
        loss.backward()
        optimizer.step()

        images, _, _ = model.renderer(model.vertices, model.faces, torch.tanh(model.textures))
        image = images.detach().cpu().numpy()[0].transpose(1, 2, 0)
        imsave(os.path.join(dir_imgs, ("frame-" + str(i) + ".png")), (255 * image).astype(np.uint8))

    # Save output igms and gif
    make_gif(os.path.join(config.PATH.OUT, "camera.gif"), dir_imgs)

    # Silhouette faces
    image = model.renderer(model.vertices, model.faces, mode='silhouettes')
    image = image.detach().cpu().numpy().transpose(1, 2, 0)
    imsave(os.path.join(config.PATH.OUT, "silhouette_camera.png"), (255 * image).astype(np.uint8))
    # Silhouette nose
    image = model.renderer(model.vertices, model.triangles_nose, mode='silhouettes')
    image = image.detach().cpu().numpy().transpose(1, 2, 0)
    imsave(os.path.join(config.PATH.OUT, "silhouette_nose.png"), (255 * image).astype(np.uint8))
    # Silhouette mouth
    image = model.renderer(model.vertices, model.triangles_mouth, mode='silhouettes')
    image = image.detach().cpu().numpy().transpose(1, 2, 0)
    imsave(os.path.join(config.PATH.OUT, "silhouette_mouth.png"), (255 * image).astype(np.uint8))

    return model


def get_optimized_model_morphing(mesh, camera, config):
    model = ModelMorphing(mesh.vertices, mesh.faces, config.CAMERA.REF_SILHOUETTE, camera.x, camera.y, camera.z)
    model.cuda()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.OPT.LR_MORPHING)

    dir_imgs = os.path.join(config.PATH.TMP, "morphing")
    if not os.path.isdir(dir_imgs):
        os.mkdir(dir_imgs)

    loop = tqdm(range(config.OPT.ITER_MORPHING))
    for i in loop:
        loop.set_description(f"Morphing model")
        optimizer.zero_grad()
        loss = model()
        loss.backward()
        optimizer.step()

        images = model.renderer(model.vertices, model.faces, mode='silhouettes')
        image = images.detach().cpu().numpy().transpose(1, 2, 0)
        imsave(os.path.join(dir_imgs, ("frame-" + str(i) + ".png")), (255 * image).astype(np.uint8))

    make_gif(os.path.join(config.PATH.OUT, "morphing.gif"), dir_imgs)

    return model


def get_optimized_model_textures(mesh, camera, config):
    if config.TEXTURES.USE_BFM_TXTS:
        base_textures = mesh.textures
    else:
        base_textures = None

    model = ModelTextures(mesh.vertices, mesh.faces, config.TEXTURES.IMG_TEXTURES, camera.x, camera.y, camera.z, base_textures)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.OPT.LR_TEXTURES, betas=(0.5, 0.999))

    loop = tqdm(range(config.OPT.ITER_TEXTURES))
    for i in loop:
        loop.set_description(f"Optimizing model textures")
        optimizer.zero_grad()
        loss = model()
        loss.backward()
        optimizer.step()

    return model
