import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from math import exp

'''
MS-SSIM Loss
'''

def gaussian(window_size, sigma):
    gauss = paddle.to_tensor([exp(-(x - window_size//2)**2 / float(2 * sigma**2)) for x in range(window_size)], dtype=paddle.float32)
    return gauss / paddle.sum(gauss)

def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(0)
    _2D_window = paddle.matmul(_1D_window, _1D_window, transpose_y=True).unsqueeze(0).unsqueeze(0)
    window = paddle.tile(_2D_window, repeat_times=[channel, 1, 1, 1])
    return window

import paddle
import paddle.nn.functional as F
from paddle import to_tensor

def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if paddle.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if paddle.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    _, channel, height, width = img1.shape
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).astype(paddle.float32).to(img1.place)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = paddle.pow(mu1, 2)
    mu2_sq = paddle.pow(mu2, 2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = paddle.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = paddle.mean(ssim_map)
    else:
        ret = paddle.mean(paddle.mean(paddle.mean(ssim_map, axis=1), axis=1), axis=1)

    if full:
        return ret, cs
    return ret

def msssim(img1, img2, window_size=11, size_average=True, val_range=None, normalize=False):
    device = img1.place
    weights = paddle.to_tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333], dtype=paddle.float32).to(device)
    levels = weights.shape[0]
    mssim = []
    mcs = []
    for _ in range(levels):
        sim, cs = ssim(img1, img2, window_size=window_size, size_average=size_average, full=True, val_range=val_range)
        mssim.append(sim)
        mcs.append(cs)

        img1 = F.avg_pool2d(img1, (2, 2))
        img2 = F.avg_pool2d(img2, (2, 2))

    mssim = paddle.stack(mssim)
    mcs = paddle.stack(mcs)

    # Normalize (to avoid NaNs during training unstable models, not compliant with original definition)
    if normalize:
        mssim = (mssim + 1) / 2
        mcs = (mcs + 1) / 2

    pow1 = paddle.pow(mcs, weights)
    pow2 = paddle.pow(mssim, weights)
    # From Matlab implementation https://ece.uwaterloo.ca/~z70wang/research/iwssim/
    output = paddle.prod(pow1[:-1] * pow2[-1])
    return output


class SSIM(nn.Layer):
    def __init__(self, window_size=11, size_average=True, val_range=None):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range

        # Assume 1 channel for SSIM
        self.channel = 1
        self.window = create_window(window_size)

    def forward(self, img1, img2):
        _, channel, _, _ = img1.shape

        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = create_window(self.window_size, channel).astype(img1.dtype).to(img1.place)
            self.window = window
            self.channel = channel
        return ssim(img1, img2, window=window, window_size=self.window_size, size_average=self.size_average)
    



class MSSSIM(nn.Layer):
    def __init__(self, window_size=11, size_average=True, channel=3):
        super(MSSSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel

    def forward(self, img1, img2):
        return msssim(img1, img2, window_size=self.window_size, size_average=self.size_average)

class TVLoss(nn.Layer):
    def __init__(self, TVLoss_weight=1):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.shape[0]
        h_x = x.shape[2]
        w_x = x.shape[3]
        count_h = self._tensor_size(x[:, :, 1:, :])  # Total number of differences computed for height
        count_w = self._tensor_size(x[:, :, :, 1:])  # Total number of differences computed for width
        h_tv = paddle.pow((x[:, :, 1:, :] - x[:, :, :h_x-1, :]), 2).sum()  
        w_tv = paddle.pow((x[:, :, :, 1:] - x[:, :, :, :w_x-1]), 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    def _tensor_size(self, t):
        return t.shape[1] * t.shape[2] * t.shape[3]