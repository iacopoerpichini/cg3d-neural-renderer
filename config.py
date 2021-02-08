import os
from yacs.config import CfgNode as CN

_C = CN()
_C.USE_BFM_2009 = True
_C.MORPHING = True
_C.RENDERING_ANGLE_STEP = 5

_C.PATH = CN()
_C.PATH.ROOT = os.path.dirname(os.path.realpath(__file__))
_C.PATH.DATA = os.path.join(_C.PATH.ROOT, "data")
_C.PATH.BFM = os.path.join(_C.PATH.DATA, "bfm-2017", "model2017-1_bfm_nomouth.h5")
_C.PATH.BFM_2009 = os.path.join(_C.PATH.DATA, "bfm-2009", "01_MorphableModel.mat")
_C.PATH.BFM_2009_REGIONS = os.path.join(_C.PATH.DATA, "bfm-2009", "face05_4seg.mat")
_C.PATH.INPUT = os.path.join(_C.PATH.DATA, "ex-1")
_C.PATH.OUT = os.path.join(_C.PATH.DATA, "out")
_C.PATH.TMP = os.path.join(_C.PATH.DATA, "tmp")

_C.CAMERA = CN()
_C.CAMERA.START_DISTANCE = -5
_C.CAMERA.START_ELEVATION = 0
_C.CAMERA.START_AZIMUTH = 0
_C.CAMERA.REF_SILHOUETTE = os.path.join(_C.PATH.INPUT, "silhouette.png")
_C.CAMERA.USE_ANCHOR_POINTS = True
_C.CAMERA.ANCHOR_MOUTH_IMG = os.path.join(_C.PATH.INPUT, "mouth.png")
_C.CAMERA.ANCHOR_NOSE_IMG = os.path.join(_C.PATH.INPUT, "nose.png")

_C.OPT = CN()
_C.OPT.ITER_CAMERA = 150
_C.OPT.ITER_MORPHING = 6
_C.OPT.ITER_TEXTURES = 25
_C.OPT.LR_CAMERA = 0.025
_C.OPT.LR_MORPHING = 0.1
_C.OPT.LR_TEXTURES = 1e-3
_C.OPT.IMG_TEXTURES = os.path.join(_C.PATH.INPUT, "textures.png")

_C.OUT = CN()
_C.OUT.DIR = os.path.join(_C.PATH.DATA, "output")
_C.OUT.CAMERA_GIF = os.path.join(_C.OUT.DIR, "camera.gif")
_C.OUT.CAMERA_SILHOUETTE = os.path.join(_C.OUT.DIR, "silhouette-camera.png")
_C.OUT.CAMERA_MOUTH = os.path.join(_C.OUT.DIR, "silhouette-mouth.png")
_C.OUT.CAMERA_NOSE = os.path.join(_C.OUT.DIR, "silhouette-nose.png")


def get_config_defaults():
    return _C.clone()