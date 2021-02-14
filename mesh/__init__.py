import torch

from mesh import read_bfm, read_bfm_2009


class Mesh:
    def __init__(self, vertices=None, faces=None, textures=None, regions=None):
        self.vertices = vertices
        self.faces = faces
        self.textures = textures
        self.regions = regions


def read_bfm_mesh(config):
    mesh = Mesh()
    if config.USE_BFM_2009:
        mesh.vertices, mesh.faces, mesh.textures, mesh.regions = read_bfm_2009.read_bfm_2009(config.PATH.BFM_2009, config.PATH.BFM_2009_REGIONS)
    else:
        mesh.vertices, mesh.faces, mesh.textures = read_bfm.read_vertices_and_faces_from_file(config.PATH.BFM)

    mesh.vertices = resize_vertices(mesh.vertices)
    mesh.vertices = center_in_origin(mesh.vertices)
    mesh.textures = scale_textures(mesh.textures)

    return mesh


def resize_vertices(vertices):
    scaling = torch.max(vertices)
    vertices = vertices/scaling
    return vertices


def scale_textures(textures):
    return textures/255.


def center_in_origin(vertices):
    center_of_mass = torch.mean(vertices[0], dim=0)
    vertices[0] = vertices - center_of_mass
    return vertices
