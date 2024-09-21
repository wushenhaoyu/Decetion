import os
import yaml
import glob
import json
from pathlib import Path
import time
 
import cv2
import numpy as np
import sys
import math
import paddle
from paddle.inference import Config
from paddle.inference import create_predictor
current_file_path = os.path.abspath(__file__)
current_folder = os.path.dirname(current_file_path)

# 将当前文件夹添加到 sys.path
sys.path.append(current_folder)

# 现在可以导入 deploy 下的模块
from deploy.python.infer import Detector, visualize_box_mask
from deploy.python.infer import Detector, visualize_box_mask
 

class PaddleDetection:
    def __init__(self, detection_type, device='GPU', run_mode='paddle', batch_size=1,
                 trt_min_shape=1, trt_max_shape=1280, trt_opt_shape=640,
                 trt_calib_mode=False, cpu_threads=1, enable_mkldnn=False,
                 enable_mkldnn_bfloat16=False, output_dir='output', threshold=0.5,
                 delete_shuffle_pass=False):
        self.detection_type = detection_type
        current_file_path = os.path.abspath(__file__)
        current_folder = os.path.dirname(current_file_path)
        model_dir = os.path.join(current_folder, 'output_inference' , detection_type)
        parent_folder = os.path.dirname(current_folder)
        self.grandparent_folder = os.path.dirname(parent_folder)
        self.detector = Detector(model_dir,
                                 device=device,
                                 run_mode=run_mode,
                                 batch_size=batch_size,
                                 trt_min_shape=trt_min_shape,
                                 trt_max_shape=trt_max_shape,
                                 trt_opt_shape=trt_opt_shape,
                                 trt_calib_mode=trt_calib_mode,
                                 cpu_threads=cpu_threads,
                                 enable_mkldnn=enable_mkldnn,
                                 enable_mkldnn_bfloat16=enable_mkldnn_bfloat16,
                                 output_dir=output_dir,
                                 threshold=threshold,
                                 delete_shuffle_pass=delete_shuffle_pass)
    
    def process_frame(self, frame):
        # 检测图像
        # results = self.detector.predict_image([frame[:, :, ::-1]], visual=False)  # bgr-->rgb
        # print(self.detector.det_times.info())
        results = self.detector.predict_image([frame], visual=False)

        # 可视化结果
        im = visualize_box_mask(frame, results, self.detector.pred_config.labels, self.detector.threshold)
        im = np.array(im)

        return im
    def start_video_capture(self):
        # 打开摄像头
        cap = cv2.VideoCapture(0)
        while True:
            # 读取一帧图像
            _, frame = cap.read()

            # 处理图像并获取结果
            processed_frame = self.process_frame(frame)

            # 显示图像
            cv2.imshow('Mask Detection', processed_frame)

            # 按 'q' 键退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # 释放资源
        cap.release()
        cv2.destroyAllWindows()

    def process_video(self, video_name, output_path=None):
        # 打开视频文件
        video_path = os.path.join(self.grandparent_folder, "media", video_name)
        output_path = os.path.join(self.grandparent_folder, "media", video_name+self.detection_type)
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
            processed_frame = self.process_frame(frame)

            # 显示图像
            # cv2.imshow('Processed Video', processed_frame)

            # 保存处理后的视频
            if output_path:
                out.write(processed_frame)


if __name__ == "__main__":

    mask_detection = PaddleDetection('vehicle_attribute_model')
    mask_detection.start_video_capture()

#mot_ppyoloe_l_36e_pipeline    行人属性目标检测可用
#mot_ppyoloe_s_36e_pipeline     行人检测跟踪可用
# mot_ppyoloe_s_36e_ppvehicle  多目标车辆检测加跟踪可用
    
    # PPLCNet_x1_0_person_attribute_945_infer  行人属性目标识别 报了一次错，