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
import threading
import json
from django.conf import settings
from django.shortcuts import render
import subprocess
current_directory = os.getcwd()


module_directory = os.path.join(current_directory)


sys.path.append(module_directory)
from django.http import StreamingHttpResponse
from django.http import HttpResponse
from multiprocessing import Process, Manager, Event
from haze.test_real import HazeRemover
from my_detection.paddle_infer import my_paddledetection
from dark.camera import VideoEnhancer
haze_net = None
dark_net =None
params = None
paddledetection_net = None
camera = None
def initialize(request):
    global haze_net
    global dark_net
    global paddledetection_net
    try:
        if haze_net is None:
            haze_net = HazeRemover()
            print("Haze Remover initialized.")

        if dark_net is None:
            dark_net = VideoEnhancer()
            print("Video Enhancer initialized.")

        if paddledetection_net is None:
            paddledetection_net = my_paddledetection()
            print("Vehicle License Detection initialized.")



    except Exception as e:
        print(f"Error initializing models: {e}")
        return HttpResponse("Error initializing models.", status=500)

    return HttpResponse("Models initialized and ready.")

def get_camera_frame_size(camera):
    ret, frame = camera.read()
    if ret:
        frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        return frame.shape[1], frame.shape[0]  # 返回宽度和高度
    return None

def gen_display(camera):
    """
    视频流生成器功能。
    """
    target_size = get_camera_frame_size(camera)
    while True:
        # 读取图片
        if camera is None:
            break
        ret, frame = camera.read()
        if ret:
            frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            # 将图片进行解码                
            if ret:
                frame= cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if params["haze_enabled"]:
                    frame = haze_net.haze_frame(frame)#传入RGB，传出RGB
                # print(frame.shape)
                if params["dark_enabled"]:
                    frame = dark_net.process_frame(frame)#传入RGB，传出RGB
                # print(frame.shape)
                frame = paddledetection_net.predit(frame)#传入RGB，传出BGR
                # frame= cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                ret, frame = cv2.imencode('.jpeg', frame)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame.tobytes() + b'\r\n')

                

def ConfirmParams(request):
    global params
    data = json.loads(request.body)
    params = {
        'haze_enabled': data.get('haze'),#去黑
        'dark_enabled': data.get('dark'),#去雾
        'hdr_enabled': data.get('hdr'),#hdr
        "people_detector":data.get("people_detector"),#行人检测
        "people_tracker":data.get("people_tracker"),#行人追踪
        'people_attr_detector': data.get('people_attr_detector'),#行人属性检测
        "vehicle_tracker":data.get("vehicle_tracker"),#车辆追踪
        'vehicle_detector': data.get('vehicle_detector'),#车辆检测
        'vehicle_attr_detector': data.get('vehicle_attr_detector'),#车辆属性检测
        'vehicleplate_detector': data.get('vehicleplate_detector'),#车牌检测
        'vehicle_press_detector': data.get('vehicle_press_detector'),#压线检测
        # "vehicle_invasion":data.get("vehicle_invasion")#违停检测
    }
    # 切换行人检测
    if params['people_detector'] != paddledetection_net.people_detector_isOn:
        paddledetection_net.turn_people_detector()
    # 切换行人追踪
    if params['people_tracker'] != paddledetection_net.people_tracker_isOn:
        paddledetection_net.turn_people_tracker()
    # 切换行人属性检测
    if params['people_attr_detector'] != paddledetection_net.people_attr_detector_isOn:
        paddledetection_net.turn_people_attr_detector()

    # 切换车辆检测状态
    if params['vehicle_detector'] != paddledetection_net.vehicle_detector_isOn:
        paddledetection_net.turn_vehicle_detector()

    # 切换车辆属性检测状态
    if params['vehicle_attr_detector'] != paddledetection_net.vehicle_attr_detector_isOn:
        paddledetection_net.turn_vehicle_attr_detector()

    # 切换车牌检测状态
    if params['vehicleplate_detector'] != paddledetection_net.vehicleplate_detector_isOn:
        paddledetection_net.turn_vehicleplate_detector()

    # 切换车辆压线检测状态
    if params['vehicle_press_detector'] != paddledetection_net.vehicle_press_detector_isOn:
        paddledetection_net.turn_vehicle_press_detector()


    return JsonResponse({'message': "success parms", "success": 1}, status=200)



def video(request):
    """
    视频流路由。将其放入img标记的src属性中。
    例如：<img src='https://ip:port/uri' >
    """
    # 视频流相机对象
    global camera
    if camera is not None:
        return StreamingHttpResponse(gen_display(camera), content_type='multipart/x-mixed-replace; boundary=frame')
    else:
        return JsonResponse({'status': 'Camera open,please open'})

def close_camera(request):
    """
    关闭摄像头路由。
    """
    global camera
    if camera is not None:
        camera.release()  # 释放摄像头资源
        cv2.destroyAllWindows()
        camera = None  # 清空摄像头对象
    return JsonResponse({'status': 'Camera closed'})

def open_camera(request):
    """
    关闭摄像头路由。
    """
    global camera
    if camera is  None:
        camera = cv2.VideoCapture(0)  
    return JsonResponse({'status': 'Camera open'})


TARGET_WIDTH = 640
TARGET_HEIGHT = 480

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
                
                # 使用 OpenCV 处理视频
                # cap = cv2.VideoCapture(file_path)
                # if not cap.isOpened():
                #     return JsonResponse({'message': "Error opening video"}, status=500)

                # 创建一个新的视频文件
                # output_file = os.path.join(settings.MEDIA_ROOT, 'resized_' + video.name)
                # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                # out = cv2.VideoWriter(output_file, fourcc, cap.get(cv2.CAP_PROP_FPS), (TARGET_WIDTH, TARGET_HEIGHT))

                # while True:
                #     ret, frame = cap.read()
                #     if not ret:
                #         break

                #     # 获取当前帧的尺寸
                #     h, w = frame.shape[:2]

                #     # 计算等比例缩放系数，确保图像不会超过目标尺寸
                #     scale = min(TARGET_WIDTH / w, TARGET_HEIGHT / h)
                #     new_w = int(w * scale)
                #     new_h = int(h * scale)

                #     # 缩放图像
                #     resized_frame = cv2.resize(frame, (new_w, new_h))

                #     # 创建黑色背景，并将缩放后的图像放置在中央
                #     top = (TARGET_HEIGHT - new_h) // 2
                #     bottom = TARGET_HEIGHT - new_h - top
                #     left = (TARGET_WIDTH - new_w) // 2
                #     right = TARGET_WIDTH - new_w - left
                #     padded_frame = cv2.copyMakeBorder(resized_frame, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))

                #     # 写入输出视频
                #     out.write(padded_frame)

                # cap.release()
                # out.release()

                # 删除原始视频（可选）
                # os.remove(file_path)
                # os.rename(output_file, file_path)
                # 生成视频的 URL
                threading.Thread(target=video_detection, args=(video.name,)).start()
            except Exception as e:
                return JsonResponse({'message': "Failed to process video", 'error': str(e)}, status=500)

        else:
            return JsonResponse({'message': "No video file uploaded."}, status=400)

    return JsonResponse({'message': "Success",  'success': 1}, status=200)

def video_detection(video_name):

    video_path = os.path.join(settings.MEDIA_ROOT, video_name)
    cap = cv2.VideoCapture(video_path)

    # 获取视频的基本信息
    fps = cap.get(cv2.CAP_PROP_FPS)  # 获取帧率
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 获取宽度
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 获取高度

    # 创建保存处理后的视频的文件名
    current_dir = os.getcwd()
    processed_video_path = os.path.join(current_dir, 'AIdjango', 'dist', 'assets', video_name)

    # 创建 VideoWriter 对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 mp4 编码
    out = cv2.VideoWriter(processed_video_path, fourcc, fps, (width, height))

    while True:
        # 读取一帧图像
        ret, frame = cap.read()
        if not ret:
            break
        frame= cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # 处理图像并获取结果
        if params["haze_enabled"]:
            frame = haze_net.haze_frame(frame)

        if params["dark_enabled"]:
            frame = dark_net.process_frame(frame)
        frame = paddledetection_net.predit(frame)

        # frame= cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # 将处理后的帧写入新的视频文件
        out.write(frame)

    # 释放资源
    cap.release()
    out.release()
    print(f"Processed video saved as {processed_video_path}")

# def upload_photo(request):
#     photo_url = None  # 初始化为 None，防止首次加载时报错
#     if request.method == 'POST':
#         print(request.body)
#         if 'photo' in request.FILES:
#             photo = request.FILES['photo']
#             file_path = os.path.join(settings.MEDIA_ROOT, photo.name)
#             try:
#                 # 将上传的照片保存到指定路径
#                 with open(file_path, 'wb') as f:
#                     for chunk in photo.chunks():
#                         f.write(chunk)
                
#                 # 生成照片的 URL 路径，确保文件上传目录与 MEDIA_URL 配置正确
#                 photo_url = os.path.join(settings.MEDIA_URL, photo.name)
#                 print(f"Photo uploaded to {file_path}")
                
#                 # 在后台启动照片处理线程
#                 threading.Thread(target=photo_processing, args=(photo.name,)).start()
                
#             except Exception as e:
#                 print(f"Failed to upload photo: {e}")
#         else:
#             print("No photo file uploaded.")
#     return JsonResponse({'message': "success parms", "success": 1}, status=200)


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
                
                # 使用 OpenCV 读取保存的照片
                img = cv2.imread(file_path)
                
                # # 获取当前图像的尺寸
                # h, w = img.shape[:2]

                # # 计算等比例缩放系数，确保图像不会超过目标尺寸
                # scale = min(TARGET_WIDTH / w, TARGET_HEIGHT / h)
                # new_w = int(w * scale)
                # new_h = int(h * scale)

                # # 缩放图像
                # resized_img = cv2.resize(img, (new_w, new_h))

                # # 创建黑色背景，并将缩放后的图像放置在中央
                # top = (TARGET_HEIGHT - new_h) // 2
                # bottom = TARGET_HEIGHT - new_h - top
                # left = (TARGET_WIDTH - new_w) // 2
                # right = TARGET_WIDTH - new_w - left
                # padded_img = cv2.copyMakeBorder(resized_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))

                # 保存处理后的照片
                # cv2.imwrite(file_path, padded_img)
                cv2.imwrite(file_path, img)
                # 生成照片的 URL 路径
                photo_url = os.path.join(settings.MEDIA_URL, photo.name)
                print(f"Photo uploaded and processed to {file_path}")
                
                # 在后台启动照片处理线程（如果需要）
                threading.Thread(target=photo_processing, args=(photo.name,)).start()

            except Exception as e:
                print(f"Failed to upload photo: {e}")
        else:
            print("No photo file uploaded.")

    return JsonResponse({'message': "Success", 'photo_url': photo_url, 'success': 1}, status=200)
def photo_processing(photo_name):

    photo_path = os.path.join(settings.MEDIA_ROOT, photo_name)

    # 读取照片
    img = cv2.imread(photo_path)
    if img is None:
        print(f"Failed to load image {photo_path}")
        return
    # 处理照片（假设你有 haze_net 和 paddle_detection_net 处理帧的函数）
    # if params["haze_enabled"]:
    img= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = haze_net.haze_frame(img)

    # if params["dark_enabled"]:
    # img = dark_net.process_frame(img)

    paddledetection_net.turn_vehicle_detector()
    img = paddledetection_net.predit(img)

    current_dir = os.getcwd()
    processed_photo_path = os.path.join(current_dir, 'AIdjango', 'dist', 'assets', photo_name)
    # 保存处理后的照片
    cv2.imwrite(processed_photo_path, img)
    command = [
        "python", "hdr/expand.py",
        processed_photo_path,
        "--out", "./AIdjango/media/",
        "--tone_map", "reinhard"
    ]
    # print(command)
    result = subprocess.run(command, capture_output=True, text=True)

    print(f"Processed photo saved as {processed_photo_path}")


def video_view(request):
    # 视频文件的 URL
    if request.body:
        data = json.loads(request.body)
        video_name = data.get("video_name")
    else:
        video_name = None
    
    video_url = f'/media/{video_name}'
    # return render(request, 'showvideo.html', {'video_url': video_url})
    return JsonResponse({'video_url': video_url,"success":1})

def photo_view(request):
    # 视频文件的 URL
    if request.body:
        data = json.loads(request.body)
        photo_name = data.get("photo_name")
    else:
        photo_name = None
    photo_url = f'/media/{photo_name}'
    # return render(request, 'showphoto.html', {'photo_url': photo_url})
    # delete_path = os.path.join(settings.MEDIA_ROOT,photo_name)
    # os.remove(delete_path)
    return JsonResponse({'photo_url': photo_url,"success":1})