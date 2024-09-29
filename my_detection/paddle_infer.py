import argparse
from collections import defaultdict
import copy
import os
import yaml
import glob
import json
from pathlib import Path
import time
import threading
import random
import cv2
import numpy as np
import math
import paddle
from paddle.inference import Config
import sys
from paddle.inference import create_predictor
current_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_directory)
 
from deploy.python.infer import Detector, visualize_box_mask
from deploy.pipeline.pphuman.attr_infer import AttrDetector
from deploy.pipeline.pipe_utils import crop_image_with_det, crop_image_with_mot, parse_mot_res
from deploy.pipeline.ppvehicle.vehicle_attr import VehicleAttr
from deploy.pipeline.ppvehicle.vehicle_plate import PlateRecognizer
from deploy.pipeline.ppvehicle.vehicle_pressing import VehiclePressingRecognizer
from deploy.pipeline.ppvehicle.lane_seg_infer import LaneSegPredictor
from deploy.pptracking.python.mot_sde_infer import SDE_Detector
from deploy.pptracking.python.mot.utils import flow_statistic, update_object_info
from deploy.pipeline.datacollector import DataCollector
from deploy.pptracking.python.mot.visualize import plot_tracking_dict
from visualize import visualize_attr, visualize_lane, visualize_vehicleplate, visualize_vehiclepress
from collections import deque

class FixedLengthQueue:
    def __init__(self, maxlen):
        self.queue = deque(maxlen=maxlen)

    def append(self, item):
        self.queue.append(item)

    def __len__(self):
        return len(self.queue)

    def __getitem__(self, index):
        return self.queue[index]

    def __iter__(self):
        return iter(self.queue)

    def __repr__(self):
        return repr(self.queue)

# paddle.enable_static()
current_dir = current_directory = os.getcwd()


def extract_crops(image, tlwhs_dict, obj_ids_dict, scores_dict):
    crops = []
    for cls_id in range(len(tlwhs_dict)):
        tlwhs = tlwhs_dict[cls_id]
        obj_ids = obj_ids_dict[cls_id]
        scores = scores_dict[cls_id]

        for i, tlwh in enumerate(tlwhs):
            x1, y1, w, h = tlwh
            intbox = (int(x1), int(y1), int(w), int(h))
            obj_id = int(obj_ids[i])
            score = float(scores[i]) if scores is not None else None

            # 裁剪图像
            crop = image[int(y1):int(y1 + h), int(x1):int(x1 + w)]

            # 存储裁剪结果
            crops.append({
                'crop': crop,
                'class_id': cls_id,
                'object_id': obj_id,
                'score': score,
                'crop_box': tlwh,
            })

    return crops
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
    model_dir = os.path.join(current_dir,'my_detection', 'output_inference' , 'PPLCNet_x1_0_person_attribute_945_infer')
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
                           output_dir='output'
    )
    return detector

def vehicleplate_detector_init():
    #初始化车牌识别器
    parser = argparse.ArgumentParser(description='Inference script')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use (cpu or gpu)')
    parser.add_argument('--run_modd', type=str, default='paddle', help='Run mode (e.g., paddle)')
    parser.add_argument('--cpu_threads', type=int, default=1, help='Number of CPU threads')
    parser.add_argument('--enable_mkldnn', action='store_true', help='Enable MKL-DNN')
    parser.add_argument('--enable_mkldnn_bfloat16', action='store_true', help='Enable MKL-DNN bfloat16')
    args = parser.parse_args()
    cfg = {
        'det_limit_side_len': 480,
        'det_model_dir': os.path.join(current_dir,'my_detection', 'output_inference' , 'ch_PP-OCRv3_det_infer'),
        'det_limit_type': 'max',
        'rec_image_shape': [3, 48, 320],
        'rec_batch_num': 6,
        'word_dict_path': os.path.join(current_dir,'my_detection', 'deploy','pipeline','ppvehicle','rec_word_dict.txt'),
        'basemode': "idbased",
        'rec_model_dir': os.path.join(current_dir,'my_detection', 'output_inference' , 'ch_PP-OCRv3_rec_infer')
    }
    print(args.device)
    recognizer = PlateRecognizer(args=args,cfg=cfg)
    return recognizer

def vehicle_press_detector_init():
    lane_seg_config = os.path.join(current_dir,'my_detection', 'deploy' , 'pipeline', 'config', 'lane_seg_config.yml')
    model_dir = os.path.join(current_dir,'my_detection', 'output_inference' , 'pp_lite_stdc2_bdd100k')
    #初始化车辆压线检测器
    laneseg_predictor = LaneSegPredictor(lane_seg_config=lane_seg_config, model_dir=model_dir)#实线识别
    press_recoginizer = VehiclePressingRecognizer(cfg=None) #压线

    return laneseg_predictor, press_recoginizer

def vehicle_sde_detector_init(region_type,region_polygon):
    #初始化车辆跟踪器
    model_dir = os.path.join(current_dir,'my_detection', 'output_inference' , 'mot_ppyoloe_s_36e_ppvehicle')
    detector = SDE_Detector(
        model_dir=model_dir,
        tracker_config=os.path.join(current_dir,'my_detection','deploy','pipeline','config','tracker_config.yml'),
        device='GPU',
        run_mode='paddle',
        region_type=region_type,
        region_polygon=region_polygon,
    )
    return detector

def people_sde_detector_init(region_type,region_polygon):
    #初始化人跟踪器
    model_dir = os.path.join(current_dir,'my_detection', 'output_inference' , 'mot_ppyoloe_s_36e_pipeline')
    detector = SDE_Detector(
        model_dir=model_dir,
        tracker_config=os.path.join(current_dir,'my_detection','deploy','pipeline','config','tracker_config.yml'),
        device='GPU',
        run_mode='paddle',
        region_type=region_type,
        region_polygon=region_polygon,
    )
    return detector

class my_paddledetection:
    def __init__(self,is_init=True):#isinit=True 起始初始化全部检测，False需要自己开启
        """
        初始化检测器控制变量
            people_detector_isOn                行人目标检测    
            vehicle_detector_isOn               车辆目标检测
            people_attr_detector_isOn           行人属性检测
            vehicle_attr_detector_isOn          车辆属性检测
            vehicleplate_detector_isOn          车牌检测
            vehicle_press_detector_isOn         车辆压线检测
        """
        self.people_detector_isOn = False
        self.vehicle_detector_isOn = False
        self.people_attr_detector_isOn = False
        self.vehicle_attr_detector_isOn = False
        self.vehicleplate_detector_isOn = False
        self.vehicle_press_detector_isOn = False
        self.people_tracker_isOn = False
        self.vehicle_tracker_isOn = False
        self.vehicle_invasion_detector_isOn = False
        self.updated_ids = {}
        """
        初始化检测器
            people_detector                行人目标检测    
            vehicle_detector               车辆目标检测
            people_attr_detector           行人属性检测
            vehicle_attr_detector          车辆属性检测
            vehicleplate_detector          车牌检测
            laneseg_predictor              实线检测
            press_recoginizer              压线检测
        """
        if is_init:
            self.people_detector = people_detector_init()
            self.vehicle_detector = vehicle_detector_init()
            self.people_attr_detector = people_attr_detector_init()
            self.vehicle_attr_detector = vehicle_attr_detector_init()
            # self.vehicleplate_detector = vehicleplate_detector_init()
            self.laneseg_predictor,self.press_recoginizer = vehicle_press_detector_init()
            self.vehicle_tracker = vehicle_sde_detector_init(region_type='horizontal',region_polygon=[])
            self.people_tracker = people_sde_detector_init(region_type='horizontal',region_polygon=[])
        self.frame = 0
        self.collector = DataCollector()
        self.people_queue = FixedLengthQueue(maxlen=40)
        self.vehicle_queue = FixedLengthQueue(maxlen=40)
        self.people_waitting_dealwith_queue = []
        self.vehicle_waitting_dealwith_queue = []
        self.people_waitting_dealwith_flag = False
        self.vehicle_waitting_dealwith_flag = False
    def people_detector_init(self):
        """单初始化行人检测"""
        self.people_detector = people_detector_init()
    def vehicle_detector_init(self):
        """单初始化车辆检测"""
        self.vehicle_detector = vehicle_detector_init()
    def people_attr_detector_init(self):
        """单初始化行人属性检测"""
        self.people_attr_detector = people_attr_detector_init()
    
    def vehicle_attr_detector_init(self):
        """单初始化车辆属性检测"""
        self.vehicle_attr_detector = vehicle_attr_detector_init()
        
    def vehicleplate_detector_init(self):
        """单初始化车牌检测"""
        self.vehicleplate_detector = vehicleplate_detector_init()
        
    def vehicle_press_detector_init(self):
        """单初始化车辆压线检测"""
        self.laneseg_predictor,self.press_recoginizer = vehicle_press_detector_init()
        
    def vehicle_sde_detector_init(self):
        """单初始化车辆追踪器"""
        self.vehicle_tracker = vehicle_sde_detector_init(region_type='horizontal',region_polygon=[])
        
    def people_sde_detector_init(self):
        """单初始化行人追踪器"""
        self.people_tracker = people_sde_detector_init(region_type='horizontal',region_polygon=[])
    def turn_people_detector(self):#切换行人检测
        if self.people_detector_isOn:
            self.people_detector_isOn = False
        else:
            self.people_detector_isOn = True
            self.people_tracker_isOn = False
    def turn_vehicle_detector(self):#切换车辆检测
        if self.vehicle_detector_isOn:
            self.vehicle_detector_isOn = False
        else:
            self.vehicle_detector_isOn = True
            self.vehicle_tracker_isOn = False
    
    def turn_vehicle_tracker(self):#切换车辆跟踪器
        if self.vehicle_tracker_isOn:
            self.vehicle_tracker_isOn = False
        else:
            self.vehicle_tracker_isOn = True
            self.vehicle_detector_isOn = False
    
    def turn_people_tracker(self):#切换行人跟踪器
        if self.people_tracker_isOn:
            self.people_tracker_isOn = False
        else:
            self.people_tracker_isOn = True
            self.people_detector_isOn = False
    
        
    def turn_people_attr_detector(self):#切换行人属性检测
        if self.people_attr_detector_isOn:
            self.people_attr_detector_isOn = False
        else:
            self.people_attr_detector_isOn = True
    
    def turn_vehicle_attr_detector(self):#切换车辆属性检测
        if self.vehicle_attr_detector_isOn:
            self.vehicle_attr_detector_isOn = False
        else:
            self.vehicle_attr_detector_isOn = True
            
    def turn_vehicleplate_detector(self):#切换车牌检测
        if self.vehicleplate_detector_isOn:
            self.vehicleplate_detector_isOn = False
        else:
            self.vehicleplate_detector_isOn = True
            
    def turn_vehicle_press_detector(self):#切换车辆压线检测
        if self.vehicle_press_detector_isOn:
            self.vehicle_press_detector_isOn = False
        else:
            self.vehicle_press_detector_isOn = True
            
    def turn_vehicle_invasion_detector(self):#切换车辆违停检测
        if self.vehicle_invasion_detector_isOn:
            self.vehicle_invasion_detector_isOn = False
            self.vehicle_detector_isOn = False
        else:
            self.vehicle_invasion_detector_isOn = True
            self.vehicle_detector_isOn = True

    def clear(self):#结果清空
        self.people_res = None
        self.vehicle_res = None
        self.people_attr_res = None
        self.vehicle_crops_res = None
        self.people_crops_res = None
        self.vehicle_attr_res = None
        self.vehicleplate_res = None
        self.vehiclepress_res = None
        self.lanes_res = None
    def predit(self,input):
        self.clear()
        reuse_det_result = self.frame != 0 

        if self.people_detector_isOn:#行人检测
            people_res = self.people_detector.predict_image([input],visual=False)
            self.people_res = self.people_detector.filter_box(people_res,0.5) # 过滤掉置信度小于0.5的框
        elif self.people_tracker_isOn:
            people_res = self.people_tracker.predict_image([copy.deepcopy(input)],visual=False,reuse_det_result=False)
            self.people_res = parse_mot_res(people_res)

        if self.vehicle_detector_isOn:#车辆检测
            vehicle_res = self.vehicle_detector.predict_image([input],visual=False)
            self.vehicle_res = self.vehicle_detector.filter_box(vehicle_res,0.5) # 过滤掉置信度小于0.5的框
        elif self.vehicle_tracker_isOn:
            self.entrance = None
            id_set = set()
            interval_id_set = set()
            in_id_list = list()
            out_id_list = list()
            prev_center = dict()
            self.records = list()
            height = input.shape[0]
            width = input.shape[1]
            if self.vehicle_tracker.do_entrance_counting or self.vehicle_tracker.do_break_in_counting or self.vehicle_invasion_detector_isOn:
                if self.vehicle_tracker.region_type == 'horizontal':
                    self.entrance = [0, height / 2., width, height / 2.]
                elif self.vehicle_tracker.region_type == 'vertical':
                    self.entrance = [width / 2, 0., width / 2, height]
                elif self.vehicle_tracker.region_type == 'custom':
                    self.entrance = []
                    assert len(
                        self.vehicle_tracker.region_polygon
                    ) % 2 == 0, "region_polygon should be pairs of coords points when do break_in counting."
                    assert len(
                        self.vehicle_tracker.region_polygon
                    ) > 6, 'region_type is custom, region_polygon should be at least 3 pairs of point coords.'

                    for i in range(0, len(self.region_polygon), 2):
                        self.entrance.append(
                            [self.region_polygon[i], self.region_polygon[i + 1]])
                    self.entrance.append([width, height])
                else:
                    raise ValueError("region_type:{} unsupported.".format(
                        self.region_type))
            vehicle_res = self.vehicle_tracker.predict_image([copy.deepcopy(input)],visual=False,reuse_det_result=False)
            self.vehicle_res = parse_mot_res(vehicle_res)
            boxes, scores , ids = vehicle_res[0]
            mot_result = ( 1, boxes[0], scores[0],ids[0])
            statistic = flow_statistic(
                mot_result,
                self.vehicle_tracker.secs_interval,
                self.vehicle_tracker.do_entrance_counting,
                self.vehicle_tracker.do_break_in_counting,
                self.vehicle_tracker.region_type,
                20,
                self.entrance,
                id_set,
                interval_id_set,
                in_id_list,
                out_id_list,
                prev_center,
                self.records,
                ids2names=self.vehicle_tracker.pred_config.labels
            )
            self.records = statistic['records']
            if self.vehicle_invasion_detector_isOn:
                object_in_region_info = {}
                object_in_region_info, illegal_parking_dict = update_object_info(
                        object_in_region_info, mot_result, self.vehicle_tracker.region_type,
                        self.entrance, 20, 5)
                if len(illegal_parking_dict) != 0:
                    for key, value in illegal_parking_dict.items():
                            plate = self.collector.get_carlp(key)
                            illegal_parking_dict[key]['plate'] = plate
            
            
        if self.people_attr_detector_isOn and self.people_res is not None:#行人属性检测
            if self.people_res['boxes'].size > 0:

                if self.people_detector_isOn:
                    self.people_crops_res = crop_image_with_det([input], self.people_res)
                elif self.people_tracker_isOn :
                    self.people_crops_res , _ , _ = crop_image_with_mot(input, self.people_res)
                    self.people_crops_res = [self.people_crops_res]

                for crop_res in self.people_crops_res:#把行人的小图片裁剪出来并属性预测
                    self.people_attr_res = self.people_attr_detector.predict_image(crop_res,visual=False)
        if ( self.vehicle_attr_detector_isOn or self.vehicleplate_detector_isOn )and self.vehicle_res is not None:#车辆图像裁剪
            if self.vehicle_res['boxes'].size > 0:
                if self.vehicle_detector_isOn:
                    self.vehicle_crops_res = crop_image_with_det([input], self.vehicle_res)
                elif self.vehicle_tracker_isOn:
                    self.vehicle_crops_res , _ , _ = crop_image_with_mot(input, self.vehicle_res)
                    self.vehicle_crops_res = [self.vehicle_crops_res]
        if self.vehicle_attr_detector_isOn and self.vehicle_crops_res is not None:
            for crop_res in self.vehicle_crops_res:#车辆属性预测
                self.vehicle_attr_res = self.vehicle_attr_detector.predict_image(crop_res,visual=False)
        if self.vehicleplate_detector_isOn and self.vehicle_crops_res is not None:
            if self.frame == 0:
                platelicenses = []#车牌预测
                for crop_res in self.vehicle_crops_res:
                    platelicense = self.vehicleplate_detector.get_platelicense(crop_res)
                    platelicenses.extend(platelicense['plate'])
                    self.vehicleplate_res = {'vehicleplate': platelicenses}
        if self.vehicle_press_detector_isOn and self.vehicle_crops_res is not None:#车辆压线检测
            vehicle_press_res_list = []
            lanes, direction = self.laneseg_predictor.run([input])
            if len(lanes) == 0:
                print(" no lanes!")
            self.lanes_res = {'output': lanes, 'direction': direction}
            vehicle_press_res_list = self.press_recoginizer.run(
                    lanes, self.vehicle_res)
            self.vehiclepress_res = {'output': vehicle_press_res_list}
        self.frame += 1
        if self.vehicle_tracker_isOn:
            if self.frame == 10:
                print( "trackid number: {}".format( len(self.vehicle_res['boxes'])))
        if self.frame == 10:
            self.frame = 0

        self.visualize_image(input)
        return self.im
    
    def visualize_image(self,image):
        self.im = image
        self.im = cv2.cvtColor(self.im, cv2.COLOR_BGR2RGB)
        if self.vehicle_res is not None and self.vehicle_tracker_isOn:
            if self.vehicle_res['boxes'].size > 0 :
                ids = self.vehicle_res['boxes'][:,0]
                scores = self.vehicle_res['boxes'][:,2]
                boxes = self.vehicle_res['boxes'][:,3:]
                boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
                boxes[:, 3] = boxes[:, 3] - boxes[:, 1]
                online_tlwhs = defaultdict(list)
                online_scores = defaultdict(list)
                online_ids = defaultdict(list)
                online_tlwhs[0] = boxes
                online_scores[0] = scores
                online_ids[0] = ids
                if self.vehicle_invasion_detector_isOn:
                    do_illegal_parking_recognition = True
                    self.im = plot_tracking_dict(
                        self.im,
                        1,
                        online_tlwhs,
                        online_ids,
                        online_scores,
                        frame_id=0,
                        fps=20,
                        ids2names=self.vehicle_tracker.pred_config.labels,
                        do_entrance_counting=self.vehicle_tracker.do_entrance_counting,
                        do_break_in_counting=self.vehicle_tracker.do_break_in_counting,
                        do_illegal_parking_recognition=do_illegal_parking_recognition,
                        records=self.records,
                        entrance=self.entrance,
                        center_traj=[{}]
                    )
                else:
                    do_illegal_parking_recognition = False
                    self.im = plot_tracking_dict(
                        self.im,
                        1,
                        online_tlwhs,
                        online_ids,
                        online_scores,
                        frame_id=0,
                        fps=20,
                        ids2names=self.vehicle_tracker.pred_config.labels,
                        do_entrance_counting=self.vehicle_tracker.do_entrance_counting,
                        do_break_in_counting=self.vehicle_tracker.do_break_in_counting,
                        do_illegal_parking_recognition=do_illegal_parking_recognition,
                        records=self.records,
                        entrance=self.entrance,
                        center_traj=[{}]
                    )
                selected_ids = []
                selected_ids_ = []
                # 遍历在线 ID
                for index, id in enumerate(online_ids[0]):
                    max_id = -1
                    for i in self.vehicle_queue:
                        if i['object_id'] > max_id:
                            max_id = i['object_id']
                        if i['object_id'] == id:
                            if online_tlwhs[0][index][2] > i['crop_box'][2] and online_tlwhs[0][index][3] > i['crop_box'][3]:
                                selected_ids.append([index,id])
                    if id > max_id:
                        selected_ids_.append([index,id])

                if selected_ids or selected_ids_:
                    res = extract_crops(self.im, online_tlwhs, online_ids, online_scores)
                    
                    # 更新 people_queue 中的元素
                    for index, id in enumerate(selected_ids):
                        for crop in self.vehicle_queue:
                            if crop['object_id'] == id[1]:
                                # 找到对应的裁剪结果并更新
                                crop['crop'] = res[id[0]]['crop']
                                crop['score'] = res[id[0]]['score']
                                #保存文件
                                # print('修改',id[1])
                                self.vehicle_waitting_dealwith_queue.append(res[id[0]])
                    for index , id in enumerate(selected_ids_):
                        self.vehicle_queue.append(res[id[0]])
                        print('添加',id[1])
                        self.vehicle_waitting_dealwith_queue.append(res[id[0]])
                        #保存文件
                    if self.vehicle_waitting_dealwith_queue:
                        self.vehicle_waitting_dealwith_flag = True
        if self.people_res is not None and self.people_tracker_isOn:
            
            if self.people_res['boxes'].size > 0 :
                ids = self.people_res['boxes'][:,0]
                scores = self.people_res['boxes'][:,2]
                boxes = self.people_res['boxes'][:,3:]
                boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
                boxes[:, 3] = boxes[:, 3] - boxes[:, 1]
                online_tlwhs = defaultdict(list)
                online_scores = defaultdict(list)
                online_ids = defaultdict(list)
                online_tlwhs[0] = boxes
                online_scores[0] = scores
                online_ids[0] = ids
                self.im = plot_tracking_dict(
                    self.im,
                    1,
                    online_tlwhs,
                    online_ids,
                    online_scores,
                    frame_id=0,
                    fps=20,
                    ids2names=self.people_tracker.pred_config.labels,
                    do_entrance_counting=self.people_tracker.do_entrance_counting,
                    do_break_in_counting=self.people_tracker.do_break_in_counting,
                    do_illegal_parking_recognition=False,
                    records=None,
                    entrance=None,
                    center_traj=[{}]
                )
                selected_ids = []
                selected_ids_ = []
                # 遍历在线 ID
                for index, id in enumerate(online_ids[0]):
                    max_id = -1
                    for i in self.people_queue:
                        if i['object_id'] > max_id:
                            max_id = i['object_id']
                        if i['object_id'] == id:
                            if online_tlwhs[0][index][2] > i['crop_box'][2] and online_tlwhs[0][index][3] > i['crop_box'][3]:
                                selected_ids.append([index,id])
                    if id > max_id:
                        selected_ids_.append([index,id])

                if selected_ids or selected_ids_:
                    res = extract_crops(self.im, online_tlwhs, online_ids, online_scores)
                    
                    # 更新 people_queue 中的元素
                    for index, id in enumerate(selected_ids):
                        for crop in self.people_queue:
                            if crop['object_id'] == id[1]:
                                # 找到对应的裁剪结果并更新
                                crop['crop'] = res[id[0]]['crop']
                                crop['score'] = res[id[0]]['score']
                                #保存文件
                                # print('修改',id)
                                self.people_waitting_dealwith_queue.append(res[id[0]])
                    for index , id in enumerate(selected_ids_):
                        self.people_queue.append(res[id[0]])
                        print('添加',id[1])
                        self.people_waitting_dealwith_queue.append(res[id[0]])
                        #保存文件
                    if self.people_waitting_dealwith_queue:
                        self.people_waitting_dealwith_flag = True
            else:
                self.im =  cv2.cvtColor(self.im, cv2.COLOR_BGR2RGB)
        if self.people_res is not None and self.people_detector_isOn:
            self.im = visualize_box_mask(image, self.people_res, labels=['target'],threshold=0.5)
            
        if self.vehicle_res is not None and self.vehicle_detector_isOn:
            self.im = visualize_box_mask(image, self.vehicle_res, labels=['target'],threshold=0.5)
        self.im = np.ascontiguousarray(np.copy(self.im))

        if self.people_attr_res is not None:
            people_attr_res_i = self.people_attr_res['output']
            self.im = visualize_attr(image, people_attr_res_i,self.people_res['boxes'])
        
        if self.vehicle_attr_res is not None:
            vehicle_attr_res_i = self.vehicle_attr_res['output']
            self.im = visualize_attr(image, vehicle_attr_res_i,self.vehicle_res['boxes'])
            
        if self.vehicleplate_res is not None:
            plates = self.vehicleplate_res['vehicleplate']
            vehicle_res_i = {}
            vehicle_res_i['boxes'] = self.vehicle_res['boxes']
            vehicle_res_i['boxes'][:,4:6] = vehicle_res_i['boxes'][:,4:6] - vehicle_res_i['boxes'][:,2:4]
            self.im  = visualize_vehicleplate(self.im, plates, vehicle_res_i['boxes'])
        
        if self.vehiclepress_res is not None:
            press_vehicle =  self.vehiclepress_res['output'][0]
            if len(press_vehicle) > 0:
                self.im = visualize_vehiclepress(self.im, self.vehiclepress_res['output'][0],0.5)
            self.im = np.ascontiguousarray(np.copy(self.im))
        if self.lanes_res is not None:
            lanes = self.lanes_res['output'][0]
            self.im = visualize_lane(self.im, lanes)
            self.im = np.ascontiguousarray(np.copy(self.im))

    # def people_dealwith_queue(self):
    #     if self.people_waitting_dealwith_flag:
    #         #写入行人处理逻辑
    #         for i in self.people_waitting_dealwith_queue:
    #             crop = i['crop']
    #             cls_id = i['class_id']
    #             obj_id = i['object_id']
    #             score = i['score']
    #             crop_box = i['crop_box']
    #             # 构造保存路径
    #             save_dir = 'AIdjango/dist/livedisplay'
    #             os.makedirs(save_dir, exist_ok=True)
    #             file_name = f"{obj_id}.png"
    #             if crop is not None and crop.size > 0:
    #                 cv2.imwrite(os.path.join(save_dir, file_name), crop)
    def people_dealwith_queue(self):
            if self.people_waitting_dealwith_flag:
                save_dir = 'AIdjango/dist/livedisplay'
                for i in self.people_waitting_dealwith_queue:
                    crop = i['crop']
                    obj_id = i['object_id']
                    print("obj",obj_id)
                    # 检查对象是否第一次监测或以 0.1 概率更新
                    if obj_id not in self.updated_ids:
                        self.updated_ids[obj_id] = True  # 标记为已更新
                        should_update = True
                    else:
                        should_update = random.random() < 0.05  # 0.1 的概率
                    if crop is not None and crop.size > 0 and should_update:
                        file_name = f"{obj_id}.png"
                        cv2.imwrite(os.path.join(save_dir, file_name), crop)
            self.people_waitting_dealwith_queue = []

    def vehicle_dealwith_queue(self):
        if self.vehicle_waitting_dealwith_flag:
            #写入车辆处理逻辑
            for i in self.vehicle_waitting_dealwith_queue:
                pass
            
            

        
        




"""if __name__ == "__main__":
    my_detection = my_paddledetection()
    #my_detection.turn_people_detector()
    #my_detection.turn_vehicle_attr_detector()
    #my_detection.turn_vehicleplate_detector()
    my_detection.turn_vehicle_detector()
    my_detection.turn_vehicle_press_detector()
    # 定义图像文件夹路径
    image_folder = os.path.join(current_dir,'test')
    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
    images.sort()  # 确保按照文件名顺序读取
    
    for image_name in images:
        # 读取图像
        img_path = os.path.join(image_folder, image_name)
        frame = cv2.imread(img_path)
        
        if frame is not None:
            input = frame[:, :, ::-1]
            img = my_detection.predit(input)
            
            # 显示图像
            cv2.imshow('Mask Detection', img)
            
            # 按 'q' 键退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    # 关闭所有窗口
    cv2.destroyAllWindows()    """








"""if __name__ == "__main__":
    my_detection = my_paddledetection()
    #my_detection.turn_people_tracker()
    #my_detection.turn_people_detector()
    #my_detection.turn_people_detector()
    #my_detection.turn_people_attr_detector()

    #my_detection.turn_vehicleplate_detector()
    #my_detection.turn_vehicle_press_detector()
    my_detection.turn_vehicle_tracker()
    # 定义图像文件夹路径
    image_folder = os.path.join(current_dir,'test')
    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
    images.sort()  # 确保按照文件名顺序读取
    
    for image_name in images:
        # 读取图像
        img_path = os.path.join(image_folder, image_name)
        frame = cv2.imread(img_path)
        
        if frame is not None:
            input = frame[:, :, ::-1]
            img = my_detection.predit(input)
            
            # 显示图像
            cv2.imshow('Mask Detection', img)
            
            # 按 'q' 键退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    # 关闭所有窗口
    cv2.destroyAllWindows()    """






def background_processing():
    while True:
        my_detection.people_dealwith_queue()
        time.sleep(1)
    # 启动后台线程


if __name__ == "__main__":
    my_detection = my_paddledetection()
    my_detection.turn_people_tracker()
    # my_detection.turn_people_detector()
    # my_detection.turn_people_attr_detector()
    # my_detection.turn_vehicle_detector()
    # my_detection.turn_vehicle_attr_detector()
    # my_detection.turn_vehicle_press_detector()
    # background_thread = threading.Thread(target=background_processing, daemon=True)
    # background_thread.start()
    cap = cv2.VideoCapture(0)
    while True:
        # 读取一帧图像
        _, frame = cap.read()
        input =  cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        img = my_detection.predit(input)
        # 显示图像
        img =  cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imshow('Mask Detection', img)
        
        # 按 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()

"""if __name__ == "__main__":
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
    cv2.destroyAllWindows()"""
 
 
# def pre_img(detector, frame:cv2):
#     results = detector.predict_image([frame[:, :, ::-1]], visual=False)  # bgr-->rgb
 