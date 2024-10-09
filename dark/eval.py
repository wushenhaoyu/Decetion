import argparse
import os
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity
import skimage.io as io
import numpy as np
from dataloader.data_loader import LowlightLoader
from model.IAT import IAT
import paddle

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

def process_images(data_loader):
    """
    处理数据加载器中的图像，并计算每对图像的SSIM和PSNR值。
    
    参数:
    data_loader (paddle.io.DataLoader): 数据加载器。
    
    返回:
    list: 包含每对图像的SSIM值的列表。
    list: 包含每对图像的PSNR值的列表。
    """
    ssim_list = []
    psnr_list = []

    for batch in data_loader:
        # 假设batch包含两幅图像
        image1, image2 = batch[0], batch[1]
        
        # 将PaddleTensor转换为NumPy数组
        image1 = image1.numpy().transpose((1, 2, 0))
        image2 = image2.numpy().transpose((1, 2, 0))
        
        # 计算SSIM和PSNR
        ssim_val = calculate_rgb_ssim(image1, image2)
        psnr_val = calculate_rgb_psnr(image1, image2)
        
        ssim_list.append(ssim_val)
        psnr_list.append(psnr_val)
    
    return ssim_list, psnr_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=str, default=0)
    parser.add_argument('--file_name', type=str, default=os.path.join(current_directory, 'demo_imgs', '9.png'))
    parser.add_argument('--normalize', type=bool, default=False)
    parser.add_argument('--task', type=str, default='enhance', help='Choose from exposure or enhance')
    parser.add_argument('--img_val_path', type=str, default=os.path.join(os.getcwd(), "data", "eval", "Low"))
    config = parser.parse_args()

    paddle.set_device('gpu:' + str(config.gpu_id))
    
    # Load Pre-trained Weights
    model = IAT()
    model.load_dict(paddle.load('transform_paddle.pdparams'))
    model.eval()

    # 创建数据加载器
    val_dataset = LowlightLoader(images_path=config.img_val_path, mode='test', normalize=False)
    val_loader = paddle.io.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=8)

    # 处理数据加载器中的图像
    ssim_list, psnr_list = process_images(val_loader)

    # 计算均值
    avg_ssim = np.mean(ssim_list)
    avg_psnr = np.mean(psnr_list)

    # 打印结果
    print(f"Average SSIM Value: {avg_ssim:.4f}")
    print(f"Average PSNR Value: {avg_psnr:.4f}")