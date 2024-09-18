import os
import time
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.io import DataLoader
import numpy as np
import argparse
import cv2
import h5py
from makedataset import Dataset
import paddle.optimizer as optim
from model import GNet  # 确保你的 PaddlePaddle 版本中有这个模型

from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import mean_squared_error as compare_mse
from skimage.metrics import structural_similarity as compare_ssim
# from paddle.vision.utils import save_image as imwrite  # 如果 PaddlePaddle 有类似功能

from loss import *
from paddle.vision.models import vgg16  # 替换为 PaddlePaddle 的 VGG16 实现
import math
from PIL import Image


paddle.set_device('gpu:0') 

def main():
    # 开关定义
    parser = argparse.ArgumentParser(description="network paddlepaddle")
    # 训练参数
    parser.add_argument("--epoch", type=int, default=1000, help='epoch number')
    parser.add_argument("--bs", type=int, default=16, help='batch size')
    parser.add_argument("--lr", type=float, default=1e-4, help='learning rate')
    parser.add_argument("--model", type=str, default="./checkpoint/", help='checkpoint directory')
    # 输入输出路径
    parser.add_argument("--intest", type=str, default="./input/", help='input synthetic path')
    parser.add_argument("--outest", type=str, default="./result/", help='output synthetic path')
    args = parser.parse_args()

    print("\nnetwork paddlepaddle")
    for p, v in zip(args.__dict__.keys(), args.__dict__.values()):
        print('\t{}: {}'.format(p, v))
    print('\n')
    arg = parser.parse_args()

    # 训练
    print('> Loading dataset...')

    GNet, G_optimizer, cur_epoch = load_checkpoint(args.model, args.lr)

    test(args, GNet)




# 加载模型
def load_checkpoint(checkpoint_dir, learnrate):
    Gmodel_name = 'paddle.pdparams'
    if os.path.exists(checkpoint_dir + Gmodel_name) :
        # 加载存在的模型
        Gmodel_info = paddle.load(checkpoint_dir + Gmodel_name)
        print('==> loading existing model:', checkpoint_dir + Gmodel_name)
        
        # 模型名称
        Model = GNet()
        
        # 设置设备
        device = paddle.set_device("gpu:0" if paddle.is_compiled_with_cuda() else "cpu")
        Model.to(device)

        # 创建优化器
        G_optimizer = optim.Adam(parameters=Model.parameters(), learning_rate=learnrate)
        
        # 将模型参数赋值进 net
        print(Gmodel_info)
        Model.set_dict(Gmodel_info)
        G_optimizer = optim.Adam(parameters=Model.parameters(), learning_rate=learnrate)
        # G_optimizer.set_state_dict(Gmodel_info['optimizer'])
        cur_epoch = 105
        # 如果使用多卡训练，使用 DataParallel
        if paddle.is_compiled_with_cuda() and paddle.device.get_device() == "gpu:0":
            Model = paddle.DataParallel(Model)
    else:
        # 创建模型
        Model = GNet()
        
        # 设置设备
        device = paddle.set_device("gpu:0" if paddle.is_compiled_with_cuda() else "cpu")
        Model.to(device)
        # 创建优化器
        G_optimizer = optim.Adam(parameters=Model.parameters(), learning_rate=learnrate)
        cur_epoch = 0

    return Model, G_optimizer, cur_epoch


def tensor_metric(img, imclean, model, data_range=1):#计算图像PSNR输入为Tensor

    img_cpu = img.data.cpu().numpy().astype(np.float32).transpose(0,2,3,1)
    imgclean = imclean.data.cpu().numpy().astype(np.float32).transpose(0,2,3,1)
    
    SUM = 0
    for i in range(img_cpu.shape[0]):
        if model == 'PSNR':
            SUM += compare_psnr(imgclean[i, :, :, :], img_cpu[i, :, :, :],data_range=data_range)
        elif model == 'MSE':
            SUM += compare_mse(imgclean[i, :, :, :], img_cpu[i, :, :, :])
        elif model == 'SSIM':
            SUM += compare_ssim(imgclean[i, :, :, :], img_cpu[i, :, :, :], data_range=data_range, multichannel = True)
        else:
            print('Model False!')
        
    return SUM/img_cpu.shape[0]


def upsample(x, y):
    _, _, H, W = y.shape
    return F.interpolate(x, size=(H, W), mode='bilinear')

# 定义测试函数
def test(argspar, model):
    files = os.listdir(argspar.intest) 
    a = []
    paddle.save(model.state_dict(), 'model_weights.pdparams')
    for i in range(len(files)):
        # 加载输入图像并归一化
        haze = np.array(Image.open(argspar.intest + files[i])) / 255  
        model.eval()  # 设置为评估模式

        # 在不记录梯度的情况下进行前向传播
        with paddle.no_grad():
            haze = paddle.to_tensor(haze.transpose(2, 0, 1)[np.newaxis, :, :, :], dtype='float32')

            # 测量前向传播的时间
            starttime = time.perf_counter()
            T_out, out1, out2, out = model(haze)
            endtime1 = time.perf_counter()

            # 取消注释这部分代码如果需要对输出进行上采样
            # out1 = upsample(out1, T_out)
            # out2 = upsample(out2, T_out)

            # 将最终输出保存为图像
            result = out
            imwrite(result.numpy(), argspar.outest + files[i], range=(0, 1))

            a.append(endtime1 - starttime)

            print(f'The {i} Time: {endtime1 - starttime:.4f}.')
    
    # 打印平均处理时间
    print(np.mean(np.array(a)))

# # 辅助函数 imwrite
# def imwrite(img, filepath, range=(0, 1)):
#     img = np.clip(img, range[0], range[1])
#     img = (img * 255).astype(np.uint8)
#     Image.fromarray(img.transpose(1, 2, 0)).save(filepath)


# def imwrite(img, filepath, range=(0, 1)):

#     img = np.clip(img, range[0], range[1])
#     img = (img * 255).astype(np.uint8) 

#     if img.shape[0] == 3:
#         img = img.transpose(1, 2, 0)
#     img_pil = Image.fromarray(img)
    
#     img_pil.save(filepath)
def imwrite(img, path, range=(0, 1)):
    # 将图像从 (0, 1) 归一化到 (0, 255)
    img = (img - range[0]) / (range[1] - range[0])  # 归一化到 0-1
    img = (img * 255).clip(0, 255).astype(np.uint8)  # 归一化到 0-255 并转换为 uint8
    
    # 去掉多余维度
    img = np.squeeze(img)
    
    # 如果是单通道灰度图
    if img.ndim == 2:
        img_pil = Image.fromarray(img, mode='L')  # 灰度图
    else:
        # 如果是 RGB 或多通道图像
        img = np.transpose(img, (1, 2, 0))  # 将 (C, H, W) 转换为 (H, W, C)
        img_pil = Image.fromarray(img)
    
    # 保存图像
    img_pil.save(path)



if __name__ == '__main__':
    main()
