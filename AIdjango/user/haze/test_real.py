import os
import time
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.io import DataLoader
import numpy as np
import argparse
import cv2
import sys
from PIL import Image
current_dir = os.path.dirname(os.path.abspath(__file__))

parent_dir = os.path.join(current_dir, '..')
sys.path.append(parent_dir)
sys.path.append(current_dir)
from model import GNet  # 确保你的 PaddlePaddle 版本中有这个模型
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import mean_squared_error as compare_mse
from skimage.metrics import structural_similarity as compare_ssim
import paddle.optimizer as optim

from paddle.vision.models import vgg16  # 替换为 PaddlePaddle 的 VGG16 实现
import math

paddle.set_device('gpu:0')

class HazeRemover:
    def __init__(self, model_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'paddle.pdparams')):
        self.model_path = model_path
        self.model = self.load_checkpoint()
        print("HazeRemover已经准备完毕")

    def load_checkpoint(self):
        if os.path.exists(self.model_path):
            # print(f'==> loading existing model: {self.model_path}')
            model_info = paddle.load(self.model_path)
            model = GNet()  # 确保 GNet 在你的代码中定义
            device = paddle.set_device("gpu:0" if paddle.is_compiled_with_cuda() else "cpu")
            model.to(device)

            model.set_dict(model_info)
            if paddle.is_compiled_with_cuda() and paddle.device.get_device() == "gpu:0":
                model = paddle.DataParallel(model)

        return model

    def haze_frame(self, frame):
        # haze = np.array(Image.open(argspar.intest + files[i])) / 255  
        haze = np.array(frame) / 255  
        self.model.eval()
        with paddle.no_grad():
            haze = paddle.to_tensor(haze.transpose(2, 0, 1)[np.newaxis, :, :, :], dtype='float32')
            _, _, _, out = self.model(haze)
            result = out.numpy()
            result = (result * 255).clip(0, 255).astype(np.uint8)
            
            result = np.squeeze(result)
            
            if result.ndim == 2:
                return result
            result = np.transpose(result, (1, 2, 0))
            return result
    def process_video(self, video_name, output_path=None):
        # 打开视频文件
        video_path = os.path.join(self.grandparent_folder, "media", video_name)
        output_path = os.path.join(self.grandparent_folder, "media", video_name+"_haze")
        cap = cv2.VideoCapture(video_path)
        
        # 获取视频参数
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # 设置视频写入对象
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 可以使用其他编码器，如 'MJPG'、'XVID' 等
            out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        while True:
            # 读取一帧图像
            ret, frame = cap.read()
            if not ret:
                break

            # 处理图像并获取结果
            processed_frame = self.haze_frame(frame)

            # 显示图像
            # cv2.imshow('Processed Video', processed_frame)

            # 保存处理后的视频
            if output_path:
                out.write(processed_frame)
    def haze_picture(self, argspar):
        files = os.listdir(argspar.intest) 
        a = []
        for i in range(len(files)):
            haze = np.array(Image.open(argspar.intest + files[i])) / 255  
            self.model.eval()
            with paddle.no_grad():
                haze = paddle.to_tensor(haze.transpose(2, 0, 1)[np.newaxis, :, :, :], dtype='float32')
                starttime = time.perf_counter()
                _, _, _, out = self.model(haze)
                endtime1 = time.perf_counter()
                result = out
                self.imwrite(result.numpy(), argspar.outest + files[i], range=(0, 1))
                a.append(endtime1 - starttime)
        #         print(f'The {i} Time: {endtime1 - starttime:.4f}.')
        # print(np.mean(np.array(a)))

    def imwrite(self, img, path, range=(0, 1)):
        img = (img - range[0]) / (range[1] - range[0])
        img = (img * 255).clip(0, 255).astype(np.uint8)
        img = np.squeeze(img)
        if img.ndim == 2:
            img_pil = Image.fromarray(img, mode='L')
        else:
            img = np.transpose(img, (1, 2, 0))
            img_pil = Image.fromarray(img)
        img_pil.save(path)


if __name__ == '__main__':


    haze_remover = HazeRemover()
    
    # 示例: 处理单帧图像
    frame_path = r"C:\Users\cat\Desktop\haze_source\input\HF_Google_283.png"
    frame = Image.open(frame_path)
    
    processed_frame = haze_remover.haze_frame(frame)
    if processed_frame.ndim == 3 and processed_frame.shape[2] == 3:
        processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR)
    cv2.imshow('Processed Frame', processed_frame)
    cv2.waitKey(0)  # 等待按键事件
    cv2.destroyAllWindows()
    

