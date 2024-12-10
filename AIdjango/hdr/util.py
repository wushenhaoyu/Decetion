import sys
sys.path.append(
    'D:\\data\\data\\cloud\\code\\hdr-expandnet\\paddle_project/utils')
import tool
import os
import paddle
import numpy as np
from numpy.random import uniform
import cv2


def process_path(directory, create=False):
    directory = os.path.expanduser(directory)
    directory = os.path.normpath(directory)
    directory = os.path.abspath(directory)
    if create:
        try:
            os.makedirs(directory)
        except:
            pass
    return directory


def split_path(directory):
    directory = process_path(directory)
    name, ext = os.path.splitext(os.path.basename(directory))
    return os.path.dirname(directory), name, ext


def map_range(x, low=0, high=1):
    return np.interp(x, [x.min(), x.max()], [low, high]).astype(x.dtype)


def str2bool(x):
    if x is None or x.lower() in ['no', 'false', 'f', '0']:
        return False
    else:
        return True


def cv2paddle(np_img):

    if not isinstance(np_img, np.ndarray):
        raise ValueError("Input should be a NumPy array.")
    

    rgb = np_img[:, :, (2, 1, 0)]
    

    x = paddle.to_tensor(rgb)
    

    perm_0 = list(range(x.ndim))
    perm_0[1] = 2
    perm_0[2] = 1
    x = paddle.transpose(x=x, perm=perm_0)
    
    perm_1 = list(range(x.ndim))
    perm_1[0] = 1
    perm_1[1] = 0
    x = paddle.transpose(x=x, perm=perm_1)
    
    return x


def paddle2cv(t_img):
    x = t_img.numpy()
    perm_2 = list(range(x.ndim))
    perm_2[0] = 2
    perm_2[2] = 0
    x = np.transpose(x, axes=perm_2)
    perm_3 = list(range(x.ndim))
    perm_3[0] = 1
    perm_3[1] = 0
    x = np.transpose(x, axes=perm_3)
    return x[:, :, (2, 1, 0)]


def resize(x, size):
    return cv2.resize(x, size)


class Exposure(object):

    def __init__(self, stops=0.0, gamma=1.0):
        self.stops = stops
        self.gamma = gamma

    def process(self, img):
        return np.clip(img * 2 ** self.stops, 0, 1) ** self.gamma




class BaseTMO(object):

    def __call__(self, img):
        return self.op.process(img)


class Reinhard(BaseTMO):

    def __init__(self, intensity=-1.0, light_adapt=0.8, color_adapt=0.0,
        gamma=2.0, randomize=False):
        if randomize:
            gamma = uniform(1.8, 2.2)
            intensity = uniform(-1.0, 1.0)
            light_adapt = uniform(0.8, 1.0)
            color_adapt = uniform(0.0, 0.2)
        self.op = cv2.createTonemapReinhard(gamma=gamma, intensity=
            intensity, light_adapt=light_adapt, color_adapt=color_adapt)


class Mantiuk(BaseTMO):

    def __init__(self, saturation=1.0, scale=0.75, gamma=2.0, randomize=False):
        if randomize:
            gamma = uniform(1.8, 2.2)
            scale = uniform(0.65, 0.85)
        self.op = cv2.createTonemapMantiuk(saturation=saturation, scale=
            scale, gamma=gamma)


class Drago(BaseTMO):

    def __init__(self, saturation=1.0, bias=0.85, gamma=2.0, randomize=False):
        if randomize:
            gamma = uniform(1.8, 2.2)
            bias = uniform(0.7, 0.9)
        self.op = cv2.createTonemapDrago(saturation=saturation, bias=bias,
            gamma=gamma)


class Durand(BaseTMO):

    def __init__(self, contrast=3, saturation=1.0, sigma_space=8,
        sigma_color=0.4, gamma=2.0, randomize=False):
        if randomize:
            gamma = uniform(1.8, 2.2)
            contrast = uniform(3.5)
        self.op = cv2.createTonemapDurand(contrast=contrast, saturation=
            saturation, sigma_space=sigma_space, sigma_color=sigma_color,
            gamma=gamma)


TMO_DICT = {'exposure': Exposure, 'reinhard': Reinhard, 'mantiuk': Mantiuk,
    'drago': Drago, 'durand': Durand}


def tone_map(img, tmo_name, **kwargs):
    return TMO_DICT[tmo_name](**kwargs)(img)






def create_tmo_param_from_args(opt):
    if opt.tone_map == 'exposure':
        return {k: opt.get(k) for k in ['gamma', 'stops']}
    else:
        return {}










