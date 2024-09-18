import atexit
import os
import datetime
import cv2
import sys
from django.http import HttpResponse
from multiprocessing import Process, Manager, Event
import shutil
import threading
current_dir = os.path.dirname(os.path.abspath(__file__))

# 假设 send_video.py 与当前脚本位于平级目录，找到它的父目录
parent_dir = os.path.join(current_dir, '..')
sys.path.append(parent_dir)

# 现在可以导入
from send_video import video
# Global variable to store the process
camera_process = None
stop_event = Event()



def clean_up():
    global camera_process
    if camera_process is not None:
        stop_event.set()  # Signal the process to stop
        camera_process.join()
        camera_process = None

# Register the clean_up function to be called on exit
atexit.register(clean_up)

def detect(stop_event, save_folder):
    # Initialize camera inside the process
    camera = cv2.VideoCapture(0)

    # Create "picture" folder

    frame_count = 0  # Used to name each image

    while not stop_event.is_set():
        # Read the current frame
        ret, frame = camera.read()

        if not ret:
            print("Failed to capture frame")
            break

        # Display the camera feed
        # cv2.imshow('Live Video', frame)

        # Save each frame as an image
        image_path = os.path.join(save_folder, f"{frame_count}.jpg")
        cv2.imwrite(image_path, frame)
        # print(f"Saved {image_path}")
        # cv2.imshow("camera", frame)

        frame_count += 1  # Update frame counter


    for i in range(100):
        new_image_name = f"{frame_count}.png"
        # 复制图片
        image_path = os.path.join(save_folder, f"{frame_count}.jpg")
        shutil.copy("C:\\Users\cat\Desktop\PaddleDetection\\test_image.png", image_path)
        frame_count+=1
    camera.release()
    cv2.destroyAllWindows()

def opencam(request):
    global camera_process
    global stop_event

    if request.method == "POST":
        if camera_process is not None and camera_process.is_alive():
            return HttpResponse("Camera is already running!")

        # 创建保存图片的文件夹
        picture_folder = "picture"
        if not os.path.exists(picture_folder):
            os.mkdir(picture_folder)

        # 创建带有时间戳的子文件夹
        current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        save_folder = os.path.join(picture_folder, current_time)
        os.mkdir(save_folder)

        # 清除 stop 事件
        stop_event.clear()

        # 启动摄像头检测进程，传入 save_folder
        camera_process = Process(target=detect, args=(stop_event, save_folder))
        camera_process.start()

        # 启动视频处理线程，传入 save_folder
        video_thread = threading.Thread(target=video, args=(save_folder,))
        video_thread.start()

        return HttpResponse(f"摄像头已打开并开始拍摄，图片保存到 {save_folder} ！")
    else:
        return HttpResponse("请使用POST请求来打开摄像头。")

def closecam(request):
    global camera_process
    global stop_event

    if camera_process is not None:
        stop_event.set()  # Signal the process to stop
        camera_process.join()
        camera_process = None  # Clear global variable reference

    return HttpResponse("Camera closed and resources released.")
