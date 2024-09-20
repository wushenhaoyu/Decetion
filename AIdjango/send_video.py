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

# 向服务器端实时发送视频截图
async def send_video(websocket,dir):
        # 获取最新文件夹路径
        # latest_folder = get_latest_folder()
        # print(latest_folder)
        folder = str(dir)
        i = -1
        start_time = time.time()
        if folder:
            while True:
                i+=1
                image_file =folder+"\\"+str(i)+".jpg"
                if os.path.exists(image_file):
                    start_time = time.time()
                    # print(image_file)  # 打印当前处理的图片文件路径
                    img = cv2.imread(str(image_file))
                    if img is None or img.size == 0:
                        print("Error: Image is empty")
                    else:
                        
                        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
                        result, imgencode = cv2.imencode('.jpg', img, encode_param)
                        data = np.array(imgencode)
                        img = data.tobytes()
                        # base64编码传输
                        img = base64.b64encode(img).decode()
                        await websocket.send("data:image/jpg;base64," + img)
                else:
                    i-=1
                    if start_time and time.time() - start_time < 50:
                        await asyncio.sleep(0.1)
                    elif not start_time:
                        continue
                    else:
                        break

                # 每秒钟发送一张图片
                # await asyncio.sleep(0.01)

        # 如果没有图片文件，则等待一段时间再重新检查
        else:
            await asyncio.sleep(50)




async def main_logic(dir):

    async with websockets.connect('ws://127.0.0.1:8000/ws/video/wms/') as websocket:
        await send_video(websocket,dir)
        print(1111111111111111111111111111111111111111111111111111)
        # await asyncio.sleep(100)  # 等待 10 秒后关闭
        # await websocket.close()

# asyncio.get_event_loop().run_until_complete(main_logic())
def video(dir):
    print(123123123)
    asyncio.run(main_logic(dir))