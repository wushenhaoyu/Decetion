import os
import numpy as np
import paddle
from paddle.vision.transforms import Normalize
from PIL import Image
import cv2
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))

parent_dir = os.path.join(current_dir, '..')
sys.path.append(parent_dir)

from dark.model import IAT


class VideoEnhancer:
    def __init__(self, gpu_id='0', normalize=False, task='enhance'):
        """
        初始化 VideoEnhancer 类，加载模型并设置相关参数。
        :param gpu_id: 使用的 GPU ID，默认为 '0'
        :param normalize: 是否对图像进行归一化处理，默认为 False
        :param task: 任务类型，可选 'exposure' 或 'enhance'，默认为 'enhance'
        """
        self.gpu_id = gpu_id
        self.normalize = normalize
        self.task = task

        # 设置GPU设备
        paddle.set_device('gpu:' + str(self.gpu_id))

        # 加载预训练模型
        self.model = self._load_model()

        # 定义归一化处理
        self.normalize_process = Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    def _load_model(self):
        """
        加载预训练模型
        :return: 加载后的模型
        """
        model = IAT()
        model_path = os.path.join(current_dir, 'transform_paddle.pdparams')
        model.load_dict(paddle.load(model_path))
        model.eval()  # 设置模型为评估模式
        return model

    def process_frame(self, frame):
        """
        处理单帧视频数据并返回增强后的帧
        :param frame: 输入的BGR格式的帧
        :return: 增强后的BGR格式帧
        """
        # 将帧转换为 PIL Image 格式
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img = np.asarray(img) / 255.0
        if img.shape[2] == 4:
            img = img[:, :, :3]  # 去掉 Alpha 通道

        # 转换为张量并调整形状为 (N, C, H, W)
        input_tensor = paddle.to_tensor(img).astype('float32')
        input_tensor = input_tensor.transpose((2, 0, 1)).unsqueeze(0)

        # 是否进行归一化处理
        if self.normalize:
            input_tensor = self.normalize_process(input_tensor)

        # 前向传播处理
        with paddle.no_grad():
            _, _, enhanced_img = self.model(input_tensor)

        # 转换为 (H, W, C) 格式，并恢复到 0-255 范围
        enhanced_img = enhanced_img.squeeze().transpose((1, 2, 0)).numpy()
        enhanced_img = (enhanced_img * 255).astype(np.uint8)

        # 转换为 BGR 格式以供 OpenCV 显示
        enhanced_img_bgr = cv2.cvtColor(enhanced_img, cv2.COLOR_RGB2BGR)

        return enhanced_img_bgr

    def start_camera(self):
        """
        启动摄像头并实时处理视频帧
        """
        # 打开摄像头
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("无法打开摄像头")
            return

        # 实时视频处理
        while True:
            ret, frame = cap.read()  # 读取一帧视频
            if not ret:
                break

            # 显示原始视频帧
            cv2.imshow('Original Video', frame)

            # 处理帧并显示增强后的视频帧
            enhanced_frame = self.process_frame(frame)
            cv2.imshow('Enhanced Video', enhanced_frame)

            # 按 'q' 键退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # 释放摄像头资源并关闭所有窗口
        cap.release()
        cv2.destroyAllWindows()


# 示例调用
if __name__ == "__main__":
    enhancer = VideoEnhancer(gpu_id='0', normalize=False, task='enhance')
    enhancer.start_camera()
