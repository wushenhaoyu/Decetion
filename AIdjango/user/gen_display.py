import cv2
from django.http import StreamingHttpResponse
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
import random
parent_dir = os.path.join(current_dir, '..')
sys.path.append(parent_dir)
sys.path.append(current_dir)
from haze.test_real import HazeRemover
from my_detection.paddle_infer import PaddleDetection
haze_net = None
paddle_detection_net = None
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
                frame = haze_net.haze_frame(frame)
                frame = paddle_detection_net. process_frame(frame)
                # 转换为byte类型的，存储在迭代器中
                ret, frame = cv2.imencode('.jpeg', frame)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame.tobytes() + b'\r\n')

                





def video(request):
    """
    视频流路由。将其放入img标记的src属性中。
    例如：<img src='https://ip:port/uri' >
    """
    # 视频流相机对象
    global haze_net
    global paddle_detection_net

    if haze_net is None:
        haze_net = HazeRemover()
    if paddle_detection_net is None:
        paddle_detection_net = PaddleDetection('mot_ppyoloe_l_36e_ppvehicle')
    camera = cv2.VideoCapture(0)
    # 使用流传输传输视频流
    return StreamingHttpResponse(gen_display(camera), content_type='multipart/x-mixed-replace; boundary=frame')


def initialize(request):
    global haze_net
    global paddle_detection_net

    if haze_net is None:
        haze_net = HazeRemover()
    if paddle_detection_net is None:
        paddle_detection_net = PaddleDetection('mot_ppyoloe_l_36e_ppvehicle')

    return HttpResponse("Models initialized and ready.")