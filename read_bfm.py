import h5py
import numpy as np
import torch
import neural_renderer as nr
import os

current_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(current_dir, 'data')
bfm = os.path.join(data_dir, 'model2017-1_bfm_nomouth.h5')

def read_vertices_and_faces_from_file(file_bfm, swap_column=True, textures=False):
    f = h5py.File(file_bfm, 'r')
    vertices = _get_vertices(f)
    vertices = vertices[None, :, :]
    faces = _get_faces(f, swap_column)
    faces = faces[None, :, :]

    if textures:
        texture = _get_textures(f)
        return vertices, faces, textures

    return vertices, faces, None

def _get_vertices(f_h5):
    ds_vertices = np.array(f_h5.get("shape/model/mean"))
    n_vertices = len(ds_vertices)//3
    vertices = np.empty((n_vertices, 3),np.float32)
    i = 0
    while i < n_vertices:
        vertices[i, :] = ds_vertices[i*3:(i+1)*3]
        i += 1
    vertices = torch.from_numpy(vertices).cuda()
    return vertices


def _get_faces(f_h5, swap_column=True):
    ds_faces = np.array(f_h5.get("shape/representer/cells"), np.int32).transpose()
    if swap_column:
        ds_faces[:, [1, 2]] = ds_faces[:, [2, 1]]

    ds_faces = torch.from_numpy(ds_faces).cuda()
    return ds_faces

def _get_textures(f_h5):
    ds_textures = np.array(f_h5.get('color/model/mean'))
    textures = torch.from_numpy(ds_textures).cuda()
    return textures

def read_obj(file):
    vertices, faces = nr.load_obj(file)
    vertices = vertices[None, :, :]  # [num_vertices, XYZ] -> [batch_size=1, num_vertices, XYZ]
    faces = faces[None, :, :]  # [num_faces, 3] -> [batch_size=1, num_faces, 3]
    return vertices, faces

if __name__ == '__main__':
    vertices, faces, textures = read_vertices_and_faces_from_file(bfm, swap_column=False, textures=True)
    print(vertices, faces, vertices.shape, faces.shape)
    print(vertices.dtype)
    print(faces.dtype)