import argparse
import os
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity
import skimage.io as io
import numpy as np
import paddle

from test_real import HazeRemover

current_directory = os.getcwd()

def calculate_rgb_ssim(image1, image2):
    """
    计算两幅RGB图像的SSIM值。
    
    参数:
    image1 (numpy.ndarray): 第一幅图像的数组。
    image2 (numpy.ndarray): 第二幅图像的数组。
    
    返回:
    float: 平均SSIM值。
    """
    # 确保图像大小相同
    if image1.shape != image2.shape:
        raise ValueError("Images must have the same dimensions.")
    
    # 对每个颜色通道计算SSIM
    sum = 0
    for channel in range(3):
        ssim_val = structural_similarity(
            image1[:, :, channel],
            image2[:, :, channel],
            win_size=11,
            gaussian_weights=True,
            multichannel=False,
            data_range=image1.max() - image1.min(),
            K1=0.01,
            K2=0.03,
            sigma=1.5
        )
        sum += ssim_val
    
    return sum / 3

def calculate_rgb_psnr(image1, image2):
    """
    计算两幅RGB图像的PSNR值。
    
    参数:
    image1 (numpy.ndarray): 第一幅图像的数组。
    image2 (numpy.ndarray): 第二幅图像的数组。
    
    返回:
    float: 平均PSNR值。
    """
    # 确保图像大小相同
    if image1.shape != image2.shape:
        raise ValueError("Images must have the same dimensions.")
    
    # 对每个颜色通道计算PSNR
    sum = 0
    for channel in range(3):
        psnr_val = compare_psnr(
            image1[:, :, channel],
            image2[:, :, channel],
            data_range=image1.max() - image1.min()
        )
        sum += psnr_val
    
    return sum / 3

if __name__ == "__main__":
    haze = HazeRemover()