import argparse
import os

import numpy as np
import paddle
from PIL.Image import Image
from tqdm import tqdm
import cv2

from dark.model import IAT

import os
import paddle
import cv2
import argparse
import warnings
import numpy as np
from paddle.vision.transforms import Normalize
from PIL import Image

current_directory = os.getcwd()
print(current_directory)
parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=str, default=0)
parser.add_argument('--file_name', type=str, default=os.path.join(current_directory, 'demo_imgs', '9.png'))
parser.add_argument('--normalize', type=bool, default=False)
parser.add_argument('--task', type=str, default='enhance', help='Choose from exposure or enhance')
config = parser.parse_args()

# Weights path
exposure_pretrain = r'best_Epoch_exposure.pdparams'
enhance_pretrain = r'best_Epoch_lol_v1.pdparams'

normalize_process = Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

paddle.set_device('gpu:' + str(config.gpu_id))
## Load Pre-train Weights
model = IAT()
model.load_dict(paddle.load('transform_paddle.pdparams'))
model = model.to(paddle.CUDAPlace(0))  # 将模型移动到
model.eval()

## Load Image
print(config.file_name)
img = Image.open(config.file_name)
img = np.asarray(img) / 255.0
if img.shape[2] == 4:
    img = img[:, :, :3]

input = paddle.to_tensor(img).astype('float32')
input = input.transpose((2, 0, 1)).unsqueeze(0)  # 转换为 (N, C, H, W) 形式
if config.normalize:    # False
    input = normalize_process(input)

## Forward Network
with paddle.no_grad():
    _, _, enhanced_img = model(input)

# 保存图像
enhanced_img = enhanced_img.squeeze().transpose((1, 2, 0)).numpy()  # 转换回 (H, W, C)
enhanced_img = (enhanced_img * 255).astype(np.uint8)  # 恢复到 0-255 范围
result_img = Image.fromarray(enhanced_img)
result_img.save('result.png')
