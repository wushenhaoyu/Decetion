import argparse
import os
import numpy as np
import paddle
from paddle.vision.transforms import Normalize
from PIL import Image
import cv2
from dark.model import IAT

# 配置参数
parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=str, default='0')
parser.add_argument('--normalize', type=bool, default=False)
parser.add_argument('--task', type=str, default='enhance', help='Choose from exposure or enhance')
config = parser.parse_args()

# 加载模型权重路径
exposure_pretrain = r'best_Epoch_exposure.pdparams'
enhance_pretrain = r'best_Epoch_lol_v1.pdparams'

normalize_process = Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

# 设置GPU设备
paddle.set_device('gpu:' + str(config.gpu_id))

# 加载预训练模型
model = IAT()
model.load_dict(paddle.load('transform_paddle.pdparams'))
model.eval()

# 打开摄像头
cap = cv2.VideoCapture(0)  # 0 是默认摄像头

if not cap.isOpened():
    print("无法打开摄像头")
    exit()

# 实时视频处理
while True:
    ret, frame = cap.read()  # 读取一帧视频
    if not ret:
        break

    # 显示原始视频帧
    cv2.imshow('Original Video', frame)

    # 处理帧，转换为 PIL Image 格式
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img = np.asarray(img) / 255.0
    if img.shape[2] == 4:
        img = img[:, :, :3]

    input_tensor = paddle.to_tensor(img).astype('float32')
    input_tensor = input_tensor.transpose((2, 0, 1)).unsqueeze(0)  # (N, C, H, W) 格式

    if config.normalize:
        input_tensor = normalize_process(input_tensor)

    # 前向传播处理
    with paddle.no_grad():
        _, _, enhanced_img = model(input_tensor)

    # 转换为图像并显示
    enhanced_img = enhanced_img.squeeze().transpose((1, 2, 0)).numpy()  # 转换回 (H, W, C)
    enhanced_img = (enhanced_img * 255).astype(np.uint8)  # 恢复到 0-255 范围
    enhanced_img_bgr = cv2.cvtColor(enhanced_img, cv2.COLOR_RGB2BGR)  # 转换为 BGR 格式以供 OpenCV 显示

    # 显示增强后的视频帧
    cv2.imshow('Enhanced Video', enhanced_img_bgr)

    # 按 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头资源并关闭所有窗口
cap.release()
cv2.destroyAllWindows()
