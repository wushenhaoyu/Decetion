
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np
import os
import cv2
import math
from paddle.vision.transforms import functional as F_vision
import matplotlib.pyplot as plt
from .ssim import ssim
EPS = 1e-3
PI = 22.0 / 7.0

# calculate PSNR
class PSNR(nn.Layer):
    def __init__(self, max_val=0):
        super(PSNR, self).__init__()
        base10 = paddle.log(paddle.to_tensor(10.0))
        max_val = paddle.to_tensor(max_val).astype('float32')

        self.register_buffer('base10', base10)
        self.register_buffer('max_val', 20 * paddle.log(max_val) / base10)

    def __call__(self, a, b):
        mse = paddle.mean((a.astype('float32') - b.astype('float32')) ** 2)
        if mse == 0:
            return 0
        return 10 * paddle.log10((1.0 / mse))


def adjust_learning_rate(optimizer, epoch, lr_decay=0.5):
    step = 20
    if not epoch % step and epoch > 0:
        for param_group in optimizer._parameter_list:
            param_group['learning_rate'] *= lr_decay
            print('Learning rate sets to {}.'.format(param_group['learning_rate']))
    else:
        for param_group in optimizer._parameter_list:
            print('Learning rate sets to {}.'.format(param_group['learning_rate']))


def get_dist_info():
    rank, world_size = 0, 1
    if paddle.distributed.is_initialized():
        rank = paddle.distributed.get_rank()
        world_size = paddle.distributed.get_world_size()
    return rank, world_size


def visualization(img, img_path, iteration):
    if not os.path.exists(img_path):
        os.makedirs(img_path)

    img = img.numpy()

    for i in range(img.shape[0]):
        name = str(iteration) + '_' + str(i) + '.jpg'
        print(name)
        img_single = np.transpose(img[i, :, :, :], (1, 2, 0))
        img_single = np.clip(img_single, 0, 1) * 255.0
        img_single = cv2.UMat(img_single).get()
        img_single = img_single / 255.0
        plt.imsave(os.path.join(img_path, name), img_single)


# Perceptual Loss
class LossNetwork(nn.Layer):
    def __init__(self, vgg_model):
        super(LossNetwork, self).__init__()
        self.vgg_layers = vgg_model
        self.layer_name_mapping = {
            '3': "relu1_2",
            '8': "relu2_2",
            '15': "relu3_3"
        }

    def output_features(self, x):
        output = {}
        for name, module in self.vgg_layers.named_children():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
        return list(output.values())

    def forward(self, pred_im, gt):
        loss = []
        pred_im_features = self.output_features(pred_im)
        gt_features = self.output_features(gt)
        for pred_im_feature, gt_feature in zip(pred_im_features, gt_features):
            loss.append(F.mse_loss(pred_im_feature, gt_feature))

        return sum(loss) / len(loss)


# Color Loss
class L_color(nn.Layer):
    def __init__(self):
        super(L_color, self).__init__()

    def forward(self, x):
        b, c, h, w = x.shape
        mean_rgb = paddle.mean(x, axis=[2, 3], keepdim=True)
        mr, mg, mb = paddle.split(mean_rgb, num_or_sections=3, axis=1)
        Drg = paddle.pow(mr - mg, 2)
        Drb = paddle.pow(mr - mb, 2)
        Dgb = paddle.pow(mb - mg, 2)
        k = paddle.pow(paddle.pow(Drg, 2) + paddle.pow(Drb, 2) + paddle.pow(Dgb, 2), 0.5)
        return k


def validation(model, val_loader):
    ssim = ssim()  # Paddle的SSIM
    psnr = PSNR()  # 自定义的PSNR
    ssim_list = []
    psnr_list = []

    for i, imgs in enumerate(val_loader):
        with paddle.no_grad():
            low_img, high_img = imgs[0], imgs[1]  # 取消cuda，Paddle自动处理设备
            _, _, enhanced_img = model(low_img)

        # SSIM 和 PSNR 计算
        ssim_value = ssim(enhanced_img, high_img).item()
        psnr_value = psnr(enhanced_img, high_img).item()

        # 记录 SSIM 和 PSNR
        ssim_list.append(ssim_value)
        psnr_list.append(psnr_value)

    SSIM_mean = np.mean(ssim_list)
    PSNR_mean = np.mean(psnr_list)

    print('The SSIM Value is:', SSIM_mean)
    print('The PSNR Value is:', PSNR_mean)
    return SSIM_mean, PSNR_mean
