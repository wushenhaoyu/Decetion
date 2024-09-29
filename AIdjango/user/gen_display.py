import cv2
from django.http import StreamingHttpResponse
import atexit
import os
import cv2
from datetime import datetime
import sys
from django.http import JsonResponse
from django.shortcuts import render
import os
from django.conf import settings
from django.conf import settings
import os
import mimetypes
import re
import threading
import json
from django.conf import settings
from django.shortcuts import render
import subprocess
from django.core.cache import cache
current_directory = os.getcwd()


module_directory = os.path.join(current_directory)


sys.path.append(module_directory)
from django.http import StreamingHttpResponse
from wsgiref.util import FileWrapper
from django.http import HttpResponse
from multiprocessing import Process, Manager, Event
from haze.test_real import HazeRemover
from my_detection.paddle_infer import my_paddledetection
from dark.camera import VideoEnhancer
haze_net = None
dark_net =None
params = None
isrecord = None
paddledetection_net = None
camera = None
camId = 0
save_thread =None
RecordCounter = None
from PyCameraList.camera_device import test_list_cameras, list_video_devices, list_audio_devices
def getAllCam(request):
    cameras = list_video_devices()
    return JsonResponse({'cam': cameras , "success": 1}, status=200)
def Camchoice(request):
    global camId
    global camera
    data = json.loads(request.body)
    print(data)
    if data is not None:
        camId = data.get("camId")
        if camera is not None:
            camera.release()  # 释放摄像头资源
            cv2.destroyAllWindows()
            camera = None  # 清空摄像头对象
        return JsonResponse({ "success": 1}, status=200)
    else:
        return JsonResponse({ "success": 0}, status=200)


def initialize():
    global haze_net
    global dark_net
    global paddledetection_net
    global params
    global isrecord
    global RecordCounter
    # try:
    #     if haze_net is None:
    #         haze_net = HazeRemover()
    #         print("Haze Remover initialized.")

    #     if dark_net is None:
    #         dark_net = VideoEnhancer()
    #         print("Video Enhancer initialized.")

    #     if paddledetection_net is None:
    #         paddledetection_net = my_paddledetection()
    #         print("Vehicle License Detection initialized.")
    #     if params is None:
    #         params = {
    #         'haze_enabled': False,
    #         'dark_enabled': False,#去雾
    #         'hdr_enabled': False,#hdr
    #         "people_detector":False,#行人检测
    #         "people_tracker":False,#行人追踪
    #         'people_attr_detector': False,#行人属性检测
    #         "vehicle_tracker":False,#车辆追踪
    #         'vehicle_detector': False,#车辆检测
    #         'vehicle_attr_detector': False,#车辆属性检测
    #         'vehicleplate_detector': False,#车牌检测
    #         'vehicle_press_detector': False
    #     }
    #     if isrecord is None:
    #         isrecord = False
    #     if RecordCounter is None:
    #         RecordCounter=0
    # except Exception as e:
    #     print(f"Error initializing models: {e}")
    #     return HttpResponse("Error initializing models.", status=500)

    # return HttpResponse("Models initialized and ready.")

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
    if params['vehicle_tracker'] != paddledetection_net.vehicle_tracker_isOn:
        paddledetection_net.turn_vehicle_tracker()

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


def get_camera_frame_size(camera):
    ret, frame = camera.read()
    if ret:
        frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        return frame.shape[1], frame.shape[0]  # 返回宽度和高度
    return None



def open_camera(request):
    """
    关闭摄像头路由。
    """
    global camera
    if camera is  None:
        camera = cv2.VideoCapture(camId)  
    return JsonResponse({'status': 'Camera open'})

def gen_display(camera):
    global RecordCounter
    target_size = get_camera_frame_size(camera)
    RecordCounter=0
    target_dir = os.path.join(os.getcwd(), "AIdjango", "dist", "livedisplay_record")
    items = os.listdir(target_dir)
    folders = [item for item in items if os.path.isdir(os.path.join(target_dir, item))]
    folder_count = len(folders)

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
                frame = paddledetection_net.predit(frame)#传入RGB，
                if isrecord:
                    if RecordCounter==0:
                            current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
                            save_dir = f'AIdjango/dist/livedisplay_record/{current_time}'
                            os.makedirs(save_dir, exist_ok=True)
                    save_path = os.path.join(save_dir, f"{RecordCounter}.jpg")
                    print(save_path)
                    cv2.imwrite(save_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))  # 保存为BGR格式
                frame= cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                ret, frame = cv2.imencode('.jpeg', frame)
                # 递增计数器
                RecordCounter += 1
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame.tobytes() + b'\r\n')
  


def video_record_on(request):
    global isrecord
    global RecordCounter 
    if request.method == 'POST':
        isrecord = True
        RecordCounter = 0
        return JsonResponse({'status': 'start record'})
def video_record_off(request):
    global RecordCounter
    global isrecord
    global save_thread 
    if request.method == 'POST':
        isrecord = False
        RecordCounter = 0
        save_thread = threading.Thread(target=saverecord)
        save_thread.start()
        return JsonResponse({'status': 'process finish'})

# def saverecord():
#         base_dir = 'AIdjango/dist/livedisplay_record'

#         folders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]

#         # 按时间戳排序，找到最新的文件夹
#         if folders:
#             latest_folder = max(folders, key=lambda x: datetime.strptime(x, "%Y-%m-%d-%H-%M-%S"))
#             save_photo_dir = os.path.join(base_dir, latest_folder)
#         image_files = [f for f in os.listdir(save_photo_dir) if f.endswith('.jpg') or f.endswith('.png')]
#         image_files.sort(key=lambda x: int(os.path.splitext(x)[0]))

#         # 检查是否有图像
#         if not image_files:
#             print("没有找到任何图像文件。")
#             exit()

#         # 获取第一张图像以获取宽高
#         first_image_path = os.path.join(save_photo_dir, image_files[0])
#         frame = cv2.imread(first_image_path)
#         height, width, layers = frame.shape

#         # 定义视频编写器
#         save_video_dir = os.path.join(os.getcwd(), "AIdjango", "dist", "livedisplay_record2video")
#         video_name = os.path.join(save_video_dir, latest_folder+'.avi')
#         fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 编码方式
#         video_writer = cv2.VideoWriter(video_name, fourcc, 30.0, (width, height))  # 30 FPS

#         # 读取并写入图像到视频
#         for image_file in image_files:
#             image_path = os.path.join(save_photo_dir, image_file)
#             frame = cv2.imread(image_path)
#             video_writer.write(frame)

#         # 释放视频编写器
            
#         video_writer.release()


def saverecord():
    base_dir = 'AIdjango/dist/livedisplay_record'

    folders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]

    # 按时间戳排序，找到最新的文件夹
    if folders:
        latest_folder = max(folders, key=lambda x: datetime.strptime(x, "%Y-%m-%d-%H-%M-%S"))
        save_photo_dir = os.path.join(base_dir, latest_folder)
    else:
        print("没有找到任何文件夹。")
        return

    image_files = [f for f in os.listdir(save_photo_dir) if f.endswith('.jpg') or f.endswith('.png')]
    image_files.sort(key=lambda x: int(os.path.splitext(x)[0]))

    # 检查是否有图像
    if not image_files:
        print("没有找到任何图像文件。")
        return

    # 获取第一张图像以获取宽高
    first_image_path = os.path.join(save_photo_dir, image_files[0])
    frame = cv2.imread(first_image_path)
    height, width, layers = frame.shape

    # 定义视频编写器
    save_video_dir = os.path.join(os.getcwd(), "AIdjango", "dist", "livedisplay_record2video")
    os.makedirs(save_video_dir, exist_ok=True)  # 确保输出目录存在
    video_name = os.path.join(save_video_dir, latest_folder + '.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 mp4 编码方式
    video_writer = cv2.VideoWriter(video_name, fourcc, 30.0, (width, height))  # 30 FPS

    # 读取并写入图像到视频
    for image_file in image_files:
        image_path = os.path.join(save_photo_dir, image_file)
        frame = cv2.imread(image_path)
        video_writer.write(frame)

    # 释放视频编写器
    video_writer.release()
    print(f"视频已成功保存为: {video_name}")

def stream_record_download(request):
        data = json.loads(request.body)
        video_name = data.get('name')
        file_path =  f'AIdjango/dist/livedisplay_record2video/{video_name}'
        response = StreamingHttpResponse(open(file_path, 'rb'))
        response['content_type'] = "application/octet-stream"
        response['Content-Disposition'] = 'attachment; filename=' + os.path.basename(file_path)
        return response

def stream_video_download(request):
        data = json.loads(request.body)
        video_name = data.get('name')
        # video_name = "2024-09-29-21-36-45.avi"
        file_path =  f'AIdjango/dist/UploadvideoProcess/{video_name}'
        response = StreamingHttpResponse(open(file_path, 'rb'))
        response['content_type'] = "application/octet-stream"
        response['Content-Disposition'] = 'attachment; filename=' + os.path.basename(file_path)
        return response

def stream_photo_download(request):
        data = json.loads(request.body)
        photo_name = data.get('name')
        # photo_name = "6bd979a269cb070014f1a1a71e90e364.png"
        file_path =  f'AIdjango/dist/UploadphotoProcess/{photo_name}'
        response = StreamingHttpResponse(open(file_path, 'rb'))
        response['content_type'] = "application/octet-stream"
        response['Content-Disposition'] = 'attachment; filename=' + os.path.basename(file_path)
        return response


    
def video(request):

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



def getAllRecordFile(request):
    base_dir = 'AIdjango/dist/livedisplay_record2video'
    files = [f for f in os.listdir(base_dir) if os.path.isfile(os.path.join(base_dir, f))]
    return JsonResponse({'files': files})
def getAllVideoFile(request):
    base_dir = 'AIdjango/dist/UploadvideoProcess'
    files = [f for f in os.listdir(base_dir) if os.path.isfile(os.path.join(base_dir, f))]
    return JsonResponse({'files': files})
def getAllPhotoFile(request):
    base_dir = 'AIdjango/dist/UploadphotoProcess'
    files = [f for f in os.listdir(base_dir) if os.path.isfile(os.path.join(base_dir, f))]
    return JsonResponse({'files': files})

TARGET_WIDTH = 640
TARGET_HEIGHT = 480

def upload_video(request):
    video_url = None  # 初始化为 None，防止首次加载时报错
    if request.method == 'POST':
        if 'video' in request.FILES:
            video = request.FILES['video']
            name, ext = os.path.splitext(video.name)
            file_path = f'AIdjango/dist/UploadvideoSave/{video.name}'
            videoname = video.name
            counter = 1
            while os.path.exists(file_path):
                file_path = f'AIdjango/dist/UploadvideoSave/{name}_{counter}{ext}'
                counter += 1
            videoname = os.path.basename(file_path)
            print(videoname)
            print(file_path)
            try:
                with open(file_path, 'wb') as f:
                    for chunk in video.chunks():
                        f.write(chunk)
            
                threading.Thread(target=video_detection, args=(videoname,)).start()
            except Exception as e:
                return JsonResponse({'message': "Failed to process video", 'error': str(e)}, status=500)
            return JsonResponse({'message': "upload finish",  "videoname":videoname,'success': 1}, status=200)

        else:
            return JsonResponse({'message': "No video file uploaded.", 'success': 0}, status=400)

    return JsonResponse({'message': "please use post",  'success': 0}, status=200)

def video_detection(video_name):
    current_dir = os.getcwd()
    video_path = os.path.join(current_dir, 'AIdjango', 'dist', 'UploadvideoSave', video_name)
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"视频总帧数: {total_frames}")
    cache.set(video_name, 0)
    cache.set(video_name+"total", 0)
    cache.set(video_name+"current", 0)
    # 获取视频的基本信息
    fps = cap.get(cv2.CAP_PROP_FPS)  # 获取帧率
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 获取宽度
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 获取高度

    # 创建保存处理后的视频的文件名
    
    processed_video_path = os.path.join(current_dir, 'AIdjango', 'dist', 'UploadvideoProcess', video_name)

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
        current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        cache.set(video_name+"total",  total_frames)  
        cache.set(video_name+"current",  total_frames)  
        cache.set(video_name, current_frame / total_frames * 100)  
    # 释放资源
    cap.release()
    out.release()


def get_progress(request):
    data = json.loads(request.body)
    video_name = data.get("video_name")
    if(video_name):
        progress = cache.get(video_name, 0)
        current = cache.get(video_name+"current", 0)
        total = cache.get(video_name+"total", 0)
        return JsonResponse({'progress': progress,"current":current,"total":total})
    return JsonResponse({'progress': "not process"})

def upload_photo(request):
    photo_url = None  # 初始化为 None，防止首次加载时报错
    if request.method == 'POST':
        if 'photo' in request.FILES:
            photo = request.FILES['photo']
            file_path = f'AIdjango/dist/UploadphotoSave/{photo.name}'
            name, ext = os.path.splitext(photo.name)
            counter = 1
            while os.path.exists(file_path):
                file_path = f'AIdjango/dist/UploadphotoSave/{name}_{counter}{ext}'
                counter += 1
            photoname = os.path.basename(file_path)
            try:
                # 将上传的照片保存到指定路径
                with open(file_path, 'wb') as f:
                    for chunk in photo.chunks():
                        f.write(chunk)
                
                img = cv2.imread(file_path)       
                cv2.imwrite(file_path, img)
                # 在后台启动照片处理线程（如果需要）
                threading.Thread(target=photo_processing, args=(photoname,)).start()
            except Exception as e:
                print(f"Failed to upload photo: {e}")
            return JsonResponse({'message': "sucess post",  "photoname":photoname,'success': 1}, status=200)
        else:
            print("No photo file uploaded.")

    return JsonResponse({'message': "please use post",  "photoname":'','success': 0}, status=200)

def photo_processing(photo_name):
    current_dir = os.getcwd()
    photo_path = os.path.join(current_dir, 'AIdjango', 'dist', 'UploadphotoSave', photo_name)
    photo_path_Process = os.path.join(current_dir, 'AIdjango', 'dist', 'UploadphotoProcess')
    photo_path_Process_path = os.path.join(current_dir, 'AIdjango', 'dist', 'UploadphotoProcess',photo_name)
    # 读取照片
    img = cv2.imread(photo_path)
    if img is None:
        print(f"Failed to load image {photo_path}")
        return
    # 处理照片（假设你有 haze_net 和 paddle_detection_net 处理帧的函数）
    img= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if params["haze_enabled"]:
        
        img = haze_net.haze_frame(img)

    if params["dark_enabled"]:
        img = dark_net.process_frame(img)
       # 保存处理后的照片
    
    img = paddledetection_net.predit(img)
    cv2.imwrite("hdr/"+photo_name, img)
    if params["hdr_enabled"]:
        command = [
        "python", "hdr/expand.py",
        "hdr/"+photo_name,
        "--tone_map", "reinhard"
    ]
        result=subprocess.run(command, capture_output=True, text=True)
        print(result)
    else:
        cv2.imwrite(photo_path_Process_path, img)
    delete_hdr_files(photo_path_Process)
    delete_photo_files("hdr/")
        

def delete_hdr_files(directory):
    # 遍历目录中的所有文件
    for filename in os.listdir(directory):
        if filename.endswith('.hdr'):
            file_path = os.path.join(directory, filename)
            try:
                os.remove(file_path)
                print(f"Deleted: {file_path}")
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")
def delete_photo_files(directory):
    # 遍历目录中的所有文件
    for filename in os.listdir(directory):
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif', '.hdr')
        if filename.lower().endswith(image_extensions):
            file_path = os.path.join(directory, filename)
            try:
                os.remove(file_path)
                print(f"Deleted: {file_path}")
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")


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

# def stream_video(request):
#     video_path = "AIdjango/dist/UploadvideoSave/show.mp4"  # 替换为视频文件的实际路径
#     wrapper = FileWrapper(open(video_path, 'rb'))
#     response = HttpResponse(wrapper, content_type='video/mp4')
#     response['Content-Length'] = os.path.getsize(video_path)
#     return response

def stream_video(request):
    data = json.loads(request.body)
    name = data.get("name")
    style = data.get("style")
    if style==1:
        video_path = "AIdjango/dist/livedisplay_record2video/"+name
    elif style==2:
        video_path = "AIdjango/dist/UploadvideoProcess/"+name
    elif style==3:
        video_path = "AIdjango/dist/UploadvideoSave/"+name
    print(video_path)
    range_header = request.META.get('HTTP_RANGE', '').strip()

    if range_header:
        size = os.path.getsize(video_path)
        start, end = parse_range_header(range_header, size)

        if start is None or end is None:
            return HttpResponse(status=416)

        if start >= size or end >= size:
            return HttpResponse(status=416)

        length = end - start + 1
        file = open(video_path, 'rb')
        file.seek(start)
        wrapper = FileWrapper(file)
        response = HttpResponse(wrapper, content_type='video/mp4', status=206)
        response['Content-Disposition'] = 'inline'
        response['Content-Length'] = str(length)
        response['Content-Range'] = f'bytes {start}-{end}/{size}'
        return response

    wrapper = FileWrapper(open(video_path, 'rb'))
    response = HttpResponse(wrapper, content_type='video/mp4')
    response['Content-Length'] = os.path.getsize(video_path)
    return response
def parse_range_header(range_header, size):
    if range_header:
        start, end = range_header.replace('bytes=', '').split('-')
        start = int(start) if start else 0
        end = int(end) if end else size - 1
        return start, end
    return None, None


def stream_photo(request):
    data = json.loads(request.body)
    name = data.get("name")
    style = data.get("style")
    if style==1:
        image_path = "AIdjango/dist/UploadphotoSave/"+name
    elif style==2:
        image_path = "AIdjango/dist/UploadphotoProcess/"+name
    print(image_path)
    # 定义存储照片的文件夹路径
    # image_folder = 'AIdjango/dist/livedisplay_record'  # 根据你的实际路径调整
    # image_path = os.path.join(image_folder, image_name)
    # 检查文件是否存在
    if os.path.exists(image_path):
        frame = cv2.imread(image_path)
        if frame is not None:
            # 编码为 JPEG 格式
            ret, buffer = cv2.imencode('.jpg', frame)
            return HttpResponse(buffer.tobytes(), content_type='image/jpeg')
    
    return HttpResponse(status=404)