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
from deploy.pipeline.pphuman.attr_infer import AttrDetector
from deploy.pipeline.pipe_utils import crop_image_with_det
from deploy.pipeline.ppvehicle.vehicle_attr import VehicleAttr
from deploy.pipeline.ppvehicle.vehicle_plate import PlateRecognizer
from deploy.pipeline.ppvehicle.vehicle_pressing import VehiclePressingRecognizer
from deploy.pipeline.ppvehicle.lane_seg_infer import LaneSegPredictor
from deploy.pptracking.python.mot_sde_infer import SDE_Detector
from visualize import visualize_attr
 

# paddle.enable_static()
current_dir = current_directory = os.getcwd()
def people_detector_init():
    #初始化行人检测器
    model_dir = os.path.join(current_dir,'my_detection', 'output_inference' , 'mot_ppyoloe_s_36e_pipeline')
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
    return detector

def vehicle_detector_init():
    #初始化车辆检测器
    model_dir = os.path.join(current_dir,'my_detection', 'output_inference' , 'mot_ppyoloe_s_36e_ppvehicle')
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
    return detector

def people_attr_detector_init():
    #初始化行人属性检测器
    model_dir = os.path.join(current_dir,'my_detection', 'output_inference' , 'PPHGNet_tiny_person_attribute_952_infer')
    detector = AttrDetector(model_dir,
                    device='GPU',
                    run_mode='paddle',
                    batch_size=1,
                    trt_min_shape=1,
                    trt_max_shape=1280,
                    trt_opt_shape=640,
                    trt_calib_mode=False,
                    cpu_threads=1,
                    enable_mkldnn=False,
                    output_dir='output',
                    threshold=0.5,)
    return detector

def vehicle_attr_detector_init():
    #初始化车辆属性检测器
    model_dir = os.path.join(current_dir,'my_detection', 'output_inference' , 'vehicle_attribute_model')
    detector =  VehicleAttr(model_dir,
                            device='GPU',
                            run_mode='paddle',
                            batch_size=1,
                            trt_min_shape=1,
                            trt_max_shape=1280,
                            trt_opt_shape=640,
                            trt_calib_mode=False,
                            cpu_threads=1,
                            enable_mkldnn=False,
                           output_dir='output',
                           threshold=0.5,
    )
    return detector

def vehicleplate_detector_init():
    #初始化车牌识别器
    args = {
        'device': 'GPU',
        'run_modd':'paddle',
        'cpu_threads': 1,
        'enable_mkldnn': False,
        'enable_mkldnn_bfloat16': False
    }
    cfg = {
        'det_limit_side_len': 480,
        'det_model_dir':'output_inference/ch_PP-OCRv3_det_infer/',
        'det_limit_type': 'max',
        'rec_image_shape': [3, 48, 320],
        'rec_batch_num': 6,
        'word_dict_path': 'deploy/pipeline/ppvehicle/rec_word_dict.txt',
        'basemode': "idbased",
        'rec_model_dir': 'output_inference/ch_PP-OCRv3_rec_infer/'
    }
    recognizer = PlateRecognizer(args=args,cfg=cfg)
    return recognizer

def vehicle_press_detector_init():
    #初始化车辆压线检测器
    laneseg_predictor = LaneSegPredictor('deploy/pipeline/config/lane_seg_config.yml','output_inference/pp_lite_stdc2_bdd100k/')#实线识别
    press_recoginizer = VehiclePressingRecognizer(cfg=None) #压线

    return laneseg_predictor, press_recoginizer

def vehicle_sde_detector_init():
    region_type = [0,0,0,0]#矩形
    #初始化车辆违停检测器
    model_dir = os.path.join(current_dir,'my_detection', 'output_inference' , 'mot_ppyoloe_s_36e_ppvehicle')
    detector = SDE_Detector(
        model_dir=model_dir,
        tracker_config='deploy/pipeline/config/tracker_config.yml',
        device='GPU',
        run_mode='paddle'
    )


    








if __name__ == "__main__":
    detector = people_detector_init()
    people_attr_detector = people_attr_detector_init()
    cap = cv2.VideoCapture(0)
    while True:
        # 读取一帧图像
        _, frame = cap.read()
        input = [frame[:, :, ::-1]]
        # 检测图像
        results = detector.predict_image(input, visual=False)  # bgr-->rgb
        results = detector.filter_box(results,0.5)
        crops_results = crop_image_with_det(input, results)
        attr_res_list = []
        for crop_result in crops_results:
            attr_res = people_attr_detector.predict_image(crop_result, visual=False)
            attr_res_list.append(attr_res)
        #print(results)
        #print(detector.det_times.info())

        # 可视化结果
        im = visualize_box_mask(frame, results, detector.pred_config.labels, detector.threshold)
        im = np.array(im)
        for attr_res in attr_res_list:
            im = visualize_attr(im, attr_res, results['boxes'])

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
 