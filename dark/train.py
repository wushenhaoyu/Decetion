import argparse
import os
import random
import numpy as np
import paddle
from paddle.io import DataLoader, Dataset
from dataloader.data_loader import LowlightLoader
from PIL import Image
import glob
from model.IAT import IAT
import paddle
import paddle.nn as nn
from  paddle.nn import functional as F
from paddle.optimizer import Adam
from paddle.optimizer.lr import CosineAnnealingDecay
import paddle.vision.transforms as transforms
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from paddle.vision.models import vgg16
from utils.utils import LossNetwork , validation


current_dir = os.getcwd()
# 命令行参数
parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=str, default=0)
parser.add_argument("--img_path", type=str, default=os.path.join(current_dir, "data", "train", "Low"))
parser.add_argument("--img_val_path", type=str, default=os.path.join(current_dir, "data", "eval", "Low"))
parser.add_argument("--normalize", action="store_false", help="Default Normalize in LOL training.")
parser.add_argument('--model_type', type=str, default='s')

parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--weight_decay', type=float, default=0.0001)
parser.add_argument('--pretrain_dir', type=str, default=None)

parser.add_argument('--num_epochs', type=int, default=400)
parser.add_argument('--display_iter', type=int, default=10)
parser.add_argument('--snapshots_folder', type=str, default="workdirs/snapshots_folder_lol_v1_patch")

config = parser.parse_args()

# 确保输出目录存在
if not os.path.exists(config.snapshots_folder):
    os.makedirs(config.snapshots_folder)

# 设置 Paddle 设备
paddle.set_device('gpu:' + str(config.gpu_id))

# 模型初始化
model = IAT()
model.load_dict(paddle.load('transform_paddle.pdparams'))
model = model.to(paddle.CUDAPlace(0))  # 将模型移动到GPU

# 加载预训练模型
if config.pretrain_dir is not None:
    model.set_state_dict(paddle.load(config.pretrain_dir))

# 数据设置
train_dataset = LowlightLoader(images_path=config.img_path, normalize=config.normalize)
train_loader = paddle.io.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=8)

val_dataset = LowlightLoader(images_path=config.img_val_path, mode='test', normalize=config.normalize)
val_loader = paddle.io.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=8)

# 优化器与调度器
optimizer = Adam(parameters=model.parameters(), learning_rate=config.lr, weight_decay=config.weight_decay)
scheduler = CosineAnnealingDecay(learning_rate=config.lr, T_max=config.num_epochs)

# 损失函数与VGG特征网络
vgg_model = vgg16(pretrained=True).features[:16]
vgg_model.eval()
vgg_model = vgg_model.to(paddle.CUDAPlace(0))  # 将VGG特征模型移动到GPU

# 冻结VGG特征网络的参数
for param in vgg_model.parameters():
    param.stop_gradient = True

L1_loss = nn.L1Loss()
L1_smooth_loss = F.smooth_l1_loss

loss_network = LossNetwork(vgg_model)
loss_network.eval()

ssim_high = 0
psnr_high = 0

# 训练循环
model.train()
print('######## Start IAT Training #########')

for epoch in range(config.num_epochs):
    print('Epoch:', epoch)
    for iteration, imgs in enumerate(train_loader):
        low_img, high_img = imgs[0].cuda(), imgs[1].cuda()
        
        optimizer.clear_gradients()
        mul, add, enhance_img = model(low_img)

        loss = L1_smooth_loss(enhance_img, high_img) + 0.04 * loss_network(enhance_img,high_img)

        loss.backward()
        optimizer.step()
        scheduler.step()

        if (iteration + 1) % config.display_iter == 0:
            print("Loss at iteration", iteration + 1, ":", loss.item())

    # 验证模型
    model.eval()
    PSNR_mean, SSIM_mean = validation(model, val_loader)

    # 保存日志
    with open(config.snapshots_folder + '/log.txt', 'a+') as f:
        f.write(f'Epoch {epoch}: SSIM: {SSIM_mean}, PSNR: {PSNR_mean}\n')

    # 保存最高SSIM的模型
    if SSIM_mean > ssim_high:
        ssim_high = SSIM_mean
        print('Highest SSIM so far:', ssim_high)
        paddle.save(model.state_dict(), os.path.join(config.snapshots_folder, "best_Epoch.pdparams"))
