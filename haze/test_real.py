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
import os
current_directory = os.getcwd()

# 将子文件夹路径添加到 sys.path
module_directory = os.path.join(current_directory)
sys.path.append(module_directory)


from haze.model import GNet  
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import mean_squared_error as compare_mse
from skimage.metrics import structural_similarity as compare_ssim
import paddle.optimizer as optim

from paddle.vision.models import vgg16  
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


    

if __name__ == "__main__":
    global total_time
    global frame_count
    total_time = 0
    frame_count = 0
    haze_remover = HazeRemover()
    cap = cv2.VideoCapture(0)
    start_time = time.time()
    while True:
        # 读取一帧图像
        _, frame = cap.read()
        t_start = time.time()
        img = haze_remover.haze_frame(frame)
        t_end = time.time()
        total_time += (t_end - t_start)
        frame_count += 1

        # 计算并显示帧率和平均每帧推理时间
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time
        avg_inference_time = total_time / frame_count if frame_count > 0 else 0

        # 将 FPS 和平均每帧推理时间绘制在增强后的视频帧上
        text = f"FPS: {fps:.2f}, Avg Inference Time: {avg_inference_time * 1000:.2f} ms"
        
        img = cv2.putText(img.copy(), text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # 显示图像
        cv2.imshow('Mask Detection', img)

        # 按 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()