import os
import yaml
import glob
import json
from pathlib import Path
import time
 
import cv2
import numpy as np
import math
import paddle
from paddle.inference import Config
from paddle.inference import create_predictor
 
from deploy.python.infer import Detector, visualize_box_mask
 
 
# paddle.enable_static()
current_dir = current_directory = os.getcwd()
model_dir = os.path.join(current_dir, 'output_inference' , 'mot_ppyoloe_l_36e_ppvehicle')
 
detector = Detector(model_dir,
                 device='GPU',
                 run_mode='paddle',
                 batch_size=1,
                 trt_min_shape=1,
                 trt_max_shape=1280,
                 trt_opt_shape=640,
                 trt_calib_mode=False,
                 cpu_threads=1,
                 enable_mkldnn=False,
                 enable_mkldnn_bfloat16=False,
                 output_dir='output',
                 threshold=0.5,
                 delete_shuffle_pass=False)
 
# img_path = "/home/elvis/paddle/PaddleDetection/demo/1116.png"
#frame = cv2.imread(img_path)
cap = cv2.VideoCapture(0)
while True:
    # 读取一帧图像
    _, frame = cap.read()

    # 检测图像
    results = detector.predict_image([frame[:, :, ::-1]], visual=False)  # bgr-->rgb
    print(results)
    print(detector.det_times.info())

    # 可视化结果
    im = visualize_box_mask(frame, results, detector.pred_config.labels, detector.threshold)
    im = np.array(im)

    # 显示图像
    cv2.imshow('Mask Detection', im)

    # 按 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()
 
 
# def pre_img(detector, frame:cv2):
#     results = detector.predict_image([frame[:, :, ::-1]], visual=False)  # bgr-->rgb
 