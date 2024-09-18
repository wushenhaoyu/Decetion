import asyncio
import websockets
import numpy as np
import base64
import os
import cv2
import time
from pathlib import Path

# 照片存储的根文件夹
PHOTO_FOLDER = 'picture'


# 获取最新的子文件夹路径
import asyncio
import base64
import numpy as np
import cv2
import os
from pathlib import Path
import re

# 照片存储的根文件夹
PHOTO_FOLDER = 'picture'

# 获取最新的子文件夹路径
def get_latest_folder():
    folders = sorted(Path(PHOTO_FOLDER).iterdir(), key=os.path.getmtime, reverse=True)
    if folders:
        return folders[0]
    return None

# 获取文件夹中的所有图片文件，按名称中的数字部分排序
def get_sorted_images(folder_path):
    images = list(Path(folder_path).glob('*.jpg'))
    # 自定义排序函数，提取文件名中的数字部分进行排序
    def extract_number(filename):
        match = re.search(r'(\d+)', filename)
        return int(match.group()) if match else 0

    images.sort(key=lambda x: extract_number(x.stem))
    return images

# 向服务器端实时发送视频截图
async def send_video(websocket):
    while True:
        # 获取最新文件夹路径
        latest_folder = get_latest_folder()
        if latest_folder:
            # 获取按名称中的数字部分排序的图片文件列表
            sorted_images = get_sorted_images(latest_folder)
            for image_file in sorted_images:
                # 读取图片
                print(image_file)  # 打印当前处理的图片文件路径
                img = cv2.imread(str(image_file))
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
                result, imgencode = cv2.imencode('.jpg', img, encode_param)
                data = np.array(imgencode)
                img = data.tobytes()
                # base64编码传输
                img = base64.b64encode(img).decode()

                await websocket.send("data:image/jpg;base64," + img)

                # 每秒钟发送一张图片
                # await asyncio.sleep(0.01)

        # 如果没有图片文件，则等待一段时间再重新检查
        else:
            await asyncio.sleep(5)



async def main_logic():

    async with websockets.connect('ws://127.0.0.1:8000/ws/video/wms/') as websocket:
        print(1111111111111111111)
        await send_video(websocket)
        print(22222222222222222222)


asyncio.get_event_loop().run_until_complete(main_logic())
