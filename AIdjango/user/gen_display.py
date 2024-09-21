import cv2
from django.http import StreamingHttpResponse
import atexit
import os
import datetime
import cv2
import sys
from django.http import JsonResponse
from django.shortcuts import render
import os
from django.conf import settings
from django.conf import settings
import os
import json
from django.conf import settings
from django.shortcuts import render
import subprocess

from django.http import StreamingHttpResponse
from django.http import HttpResponse
from multiprocessing import Process, Manager, Event
import shutil
import threading
current_dir = os.path.dirname(os.path.abspath(__file__))
import random
parent_dir = os.path.join(current_dir, '..')
sys.path.append(parent_dir)
sys.path.append(current_dir)
# from haze.test_real import HazeRemover
# from my_detection.paddle_infer import PaddleDetection
# from dark.camera import VideoEnhancer
haze_net = None
dark_net =None
params = None
VehicleLicense_net = None
PedestrianAttributeDetection_net = None
PedestrianAttributeRecognition_net = None
PedestrianDetectionTracking_net = None
VehicleDetectionTracking_net = None
VehicleAttribute_net = None
LaneDetection_net = None
FallDetection_net = None
def gen_display(camera):
    """
    视频流生成器功能。
    """
    while True:
        # 读取图片
        ret, frame = camera.read()
        if ret:
            frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            # 将图片进行解码                
            if ret:
                # frame= cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # frame= cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                # if params["haze_enabled"]:
                #     frame = haze_net.haze_frame(frame)

                # if params["dark_enabled"]:
                #     frame = dark_net.process_frame(frame)

                # if params["vehicle_license_enabled"]:
                #     frame = VehicleLicense_net.process_frame(frame)

                # if params["pedestrian_attribute_detection_enabled"]:
                #     frame = PedestrianAttributeDetection_net.process_frame(frame)

                # if params["pedestrian_attribute_recognition_enabled"]:
                #     frame = PedestrianAttributeRecognition_net.process_frame(frame)

                # if params["pedestrian_detection_tracking_enabled"]:
                #     frame = PedestrianDetectionTracking_net.process_frame(frame)

                # if params["vehicle_detection_tracking_enabled"]:
                #     frame = VehicleDetectionTracking_net.process_frame(frame)

                # if params["vehicle_attribute_enabled"]:
                #     frame = VehicleAttribute_net.process_frame(frame)

                # if params["lane_detection_enabled"]:
                #     frame = LaneDetection_net.process_frame(frame)

                # if params["fall_detection_enabled"]:
                #     frame = FallDetection_net.process_frame(frame)
                # 转换为byte类型的，存储在迭代器中
                ret, frame = cv2.imencode('.jpeg', frame)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame.tobytes() + b'\r\n')

                

def ConfirmParams(request):
    global params
    data = json.loads(request.body)
    print(data)
    params = {
        'haze_enabled': data.get('haze'),
        'dark_enabled': data.get('dark'),
        'vehicle_license_enabled': data.get('vehicle_license'),
        'pedestrian_attribute_detection_enabled': data.get('pedestrian_attribute_detection'),
        'pedestrian_attribute_recognition_enabled': data.get('pedestrian_attribute_recognition'),
        'pedestrian_detection_tracking_enabled': data.get('pedestrian_detection_tracking'),
        'vehicle_detection_tracking_enabled': data.get('vehicle_detection_tracking'),
        'vehicle_attribute_enabled': data.get('vehicle_attribute'),
        'lane_detection_enabled': data.get('lane_detection'),
        'fall_detection_enabled': data.get('fall_detection'),
    }
    return JsonResponse({'message': "success parms", "success": 1}, status=200)



def video(request):
    """
    视频流路由。将其放入img标记的src属性中。
    例如：<img src='https://ip:port/uri' >
    """
    # 视频流相机对象

    camera = cv2.VideoCapture(0)
    # 使用流传输传输视频流
    return StreamingHttpResponse(gen_display(camera), content_type='multipart/x-mixed-replace; boundary=frame')


# def initialize(request):
#     global haze_net
#     global dark_net
#     global VehicleLicense_net
#     global PedestrianAttributeDetection_net 
#     global PedestrianAttributeRecognition_net 
#     global PedestrianDetectionTracking_net
#     global VehicleDetectionTracking_net 
#     global VehicleAttribute_net 
#     global LaneDetection_net 
#     global FallDetection_net 
#     try:
#         if haze_net is None:
#             haze_net = HazeRemover()
#             print("Haze Remover initialized.")

#         if dark_net is None:
#             dark_net = VideoEnhancer()
#             print("Video Enhancer initialized.")

#         if VehicleLicense_net is None:
#             VehicleLicense_net = PaddleDetection('ch_PP-OCRv3_det_infer')##车牌检查
#             print("Vehicle License Detection initialized.")

#         if PedestrianAttributeDetection_net is None:
#             PedestrianAttributeDetection_net = PaddleDetection('mot_ppyoloe_l_36e_pipeline')#行人属性目标检测
#             print("Pedestrian Attribute Detection initialized.")

#         if PedestrianDetectionTracking_net is None:
#             PedestrianDetectionTracking_net = PaddleDetection('mot_ppyoloe_s_36e_pipeline')#行人检测与跟踪
#             print("Pedestrian Detection Tracking initialized.")

#         if VehicleDetectionTracking_net is None:
#             VehicleDetectionTracking_net = PaddleDetection('ch_PP-OCRv3_det_infer')##多目标车辆检测与跟踪
#             print("Vehicle Detection Tracking initialized.")

#         if LaneDetection_net is None:
#             LaneDetection_net = PaddleDetection('pp_lite_stdc2_bdd100k')#车道检测
#             print("Lane Detection initialized.")

#         if PedestrianAttributeRecognition_net is None:
#             PedestrianAttributeRecognition_net = PaddleDetection('PPLCNet_x1_0_person_attribute_945_infer')#行人属性目标识别
#             print("Pedestrian Attribute Recognition initialized.")


#         if FallDetection_net is None:
#             FallDetection_net = PaddleDetection('STGCN')#摔倒检测
#             print("Fall Detection initialized.")


#         if VehicleAttribute_net is None:
#             VehicleAttribute_net = PaddleDetection('vehicle_attribute_model')#车辆属性
#             print("Vehicle Attribute Detection initialized.")


#     except Exception as e:
#         print(11111111111111111)
#         print(f"Error initializing models: {e}")
#         return HttpResponse("Error initializing models.", status=500)

#     return HttpResponse("Models initialized and ready.")




def upload_video(request):
    video_url = None  # 初始化为 None，防止首次加载时报错
    if request.method == 'POST':
        if 'video' in request.FILES:
            video = request.FILES['video']
            file_path = os.path.join(settings.MEDIA_ROOT, video.name)
            try:
                with open(file_path, 'wb') as f:
                    for chunk in video.chunks():
                        f.write(chunk)
                # 生成视频的 URL 路径，确保文件上传目录与 MEDIA_URL 配置正确
                video_url = os.path.join(settings.MEDIA_URL, video.name)
                print(f"Video uploaded to {file_path}")
                # threading.Thread(target=video_detection, args=(video.name,)).start()
            except Exception as e:
                print(f"Failed to upload video: {e}")
        else:
            print("No video file uploaded.")
    return JsonResponse({'message': "success parms", "success": 1}, status=200)



# def video_detection(video_name):

#     video_path = os.path.join(settings.MEDIA_ROOT, video_name)
#     cap = cv2.VideoCapture(video_path)

#     # 获取视频的基本信息
#     fps = cap.get(cv2.CAP_PROP_FPS)  # 获取帧率
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 获取宽度
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 获取高度

#     # 创建保存处理后的视频的文件名
#     base_name = os.path.splitext(video_name)[0]
#     processed_video_name = f"{base_name}-processed.mp4"
#     processed_video_path = os.path.join(settings.MEDIA_ROOT, processed_video_name)

#     # 创建 VideoWriter 对象
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 mp4 编码
#     out = cv2.VideoWriter(processed_video_path, fourcc, fps, (width, height))

#     while True:
#         # 读取一帧图像
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # 处理图像并获取结果
#         if params["haze_enabled"]:
#             frame = haze_net.haze_frame(frame)

#         if params["dark_enabled"]:
#             frame = dark_net.process_frame(frame)

#         if params["vehicle_license_enabled"]:
#             frame = VehicleLicense_net.process_frame(frame)

#         if params["pedestrian_attribute_detection_enabled"]:
#             frame = PedestrianAttributeDetection_net.process_frame(frame)

#         if params["pedestrian_attribute_recognition_enabled"]:
#             frame = PedestrianAttributeRecognition_net.process_frame(frame)

#         if params["pedestrian_detection_tracking_enabled"]:
#             frame = PedestrianDetectionTracking_net.process_frame(frame)

#         if params["vehicle_detection_tracking_enabled"]:
#             frame = VehicleDetectionTracking_net.process_frame(frame)

#         if params["vehicle_attribute_enabled"]:
#             frame = VehicleAttribute_net.process_frame(frame)

#         if params["lane_detection_enabled"]:
#             frame = LaneDetection_net.process_frame(frame)

#         if params["fall_detection_enabled"]:
#             frame = FallDetection_net.process_frame(frame)

#         # 将处理后的帧写入新的视频文件
#         out.write(frame)

#     # 释放资源
#     cap.release()
#     out.release()
#     print(f"Processed video saved as {processed_video_path}")

def upload_photo(request):
    photo_url = None  # 初始化为 None，防止首次加载时报错
    if request.method == 'POST':
        if 'photo' in request.FILES:
            photo = request.FILES['photo']
            file_path = os.path.join(settings.MEDIA_ROOT, photo.name)
            try:
                # 将上传的照片保存到指定路径
                with open(file_path, 'wb') as f:
                    for chunk in photo.chunks():
                        f.write(chunk)
                
                # 生成照片的 URL 路径，确保文件上传目录与 MEDIA_URL 配置正确
                photo_url = os.path.join(settings.MEDIA_URL, photo.name)
                print(f"Photo uploaded to {file_path}")
                
                # 在后台启动照片处理线程
                # threading.Thread(target=photo_processing, args=(photo.name,)).start()
                
            except Exception as e:
                print(f"Failed to upload photo: {e}")
        else:
            print("No photo file uploaded.")
    return JsonResponse({'message': "success parms", "success": 1}, status=200)

# def photo_processing(photo_name):

#     photo_path = os.path.join(settings.MEDIA_ROOT, photo_name)

#     # 读取照片
#     img = cv2.imread(photo_path)
#     if img is None:
#         print(f"Failed to load image {photo_path}")
#         return

#     # 处理照片（假设你有 haze_net 和 paddle_detection_net 处理帧的函数）
#     if params["haze_enabled"]:
#         frame = haze_net.haze_frame(frame)

#     if params["dark_enabled"]:
#         frame = dark_net.process_frame(frame)

#     if params["vehicle_license_enabled"]:
#         frame = VehicleLicense_net.process_frame(frame)

#     if params["pedestrian_attribute_detection_enabled"]:
#             rame = PedestrianAttributeDetection_net.process_frame(frame)

#     if params["pedestrian_attribute_recognition_enabled"]:
#         frame = PedestrianAttributeRecognition_net.process_frame(frame)

#     if params["pedestrian_detection_tracking_enabled"]:
#         frame = PedestrianDetectionTracking_net.process_frame(frame)

#     if params["vehicle_detection_tracking_enabled"]:
#         frame = VehicleDetectionTracking_net.process_frame(frame)

#     if params["vehicle_attribute_enabled"]:
#         frame = VehicleAttribute_net.process_frame(frame)

#     if params["lane_detection_enabled"]:
#         frame = LaneDetection_net.process_frame(frame)

#     if params["fall_detection_enabled"]:
#         frame = FallDetection_net.process_frame(frame)

#     # 创建保存处理后照片的文件名
#     base_name = os.path.splitext(photo_name)[0]
#     processed_photo_name = f"{base_name}-processed.jpg"
#     processed_photo_path = os.path.join(settings.MEDIA_ROOT, processed_photo_name)

#     # 保存处理后的照片
#     cv2.imwrite(processed_photo_path, img)

#     print(f"Processed photo saved as {processed_photo_path}")


def video_view(request):
    # 视频文件的 URL
    if request.body:
        data = json.loads(request.body)
        video_name = data.get(video_name)
    else:
        video_name = None
    
    video_url = f'/media/{video_name}'
    # return render(request, 'showvideo.html', {'video_url': video_url})
    return JsonResponse({'video_url': video_url,"success":1})

def photo_view(request):
    # 视频文件的 URL
    if request.body:
        data = json.loads(request.body)
        photo_name = data.get(photo_name)
    else:
        photo_name = None
    photo_url = f'/media/{photo_name}'
    # return render(request, 'showphoto.html', {'photo_url': photo_url})
    return JsonResponse({'photo_url': photo_url,"success":1})