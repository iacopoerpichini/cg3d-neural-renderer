import os
from enum import IntEnum

import torch
import numpy as np
import scipy.io

current_dir = "/tmp/pycharm_project_809/data/out"#os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(current_dir, '../data', "bfm-2009")
bfm_2009 = os.path.join(data_dir, '01_MorphableModel.mat')
regions_file = os.path.join(data_dir, "face05_4seg.mat")


class RegionType(IntEnum):
    NOSE = 0
    EYES = 1
    MOUTH = 2
    FACE = 3


def read_bfm_2009(file_model, file_regions):
    f = scipy.io.loadmat(file_model)
    ds_vertices = np.array(f.get("shapeMU"))
    ds_textures = np.array(f.get("texMU"))
    faces = np.array(f.get("tl"), np.int32) - 1

    # Rad vertices and textures of the model
    n_vertices = len(ds_vertices) // 3
    vertices = np.empty((n_vertices, 3), np.float32)
    textures = np.empty((n_vertices, 3), np.float32)
    i = 0
    while i < n_vertices:
        vertices[i, :] = ds_vertices[i * 3:(i + 1) * 3, 0]
        textures[i, :] = ds_textures[i * 3:(i + 1) * 3, 0]
        i += 1

    # Convert vertex textures to face textures
    textures = read_textures(faces, textures)

    # Read face regions
    f_regions = scipy.io.loadmat(file_regions)
    regions = f_regions.get("face05_4seg")[:, 0]

    vertices = vertices[None, :, :]
    faces = faces[None, :, :]
    textures = textures[None, :]

    vertices = torch.from_numpy(vertices).cuda()
    faces = torch.from_numpy(faces).cuda()
    textures = torch.from_numpy(textures).cuda()

    return vertices, faces, textures, regions


def read_textures(faces, v_textures):
    texture_size = 2
    textures = np.ones((faces.shape[0], texture_size, texture_size, texture_size, 3), dtype=np.float32)
    for i in range(faces.shape[0]):
        v1, v2, v3 = faces[i]
        textures[i, :, :, :] = (v_textures[v1] + v_textures[v2] + v_textures[v3])/3.

    return textures


def filter_region(vertices, triangles, regions, region_type):
    mask = torch.tensor((regions == int(region_type)).nonzero()[0], dtype=torch.int).cuda()
    list_triangles = [t.detach().cpu().numpy() for t in triangles[0] if t[0] in mask and t[1] in mask and t[2] in mask]
    triangles = np.empty((1, len(list_triangles), 3), dtype=np.int32)
    for i in range(len(list_triangles)):
        triangles[:, i, :] = list_triangles[i]
    triangles = torch.tensor(triangles).cuda()

    return vertices, triangles
