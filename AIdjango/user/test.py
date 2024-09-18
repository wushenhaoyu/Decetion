import cv2
import os
import datetime

def detect():
    # 创建"picture"文件夹
    picture_folder = "picture"
    if not os.path.exists(picture_folder):
        os.mkdir(picture_folder)

    # 以当前时间命名创建一个子文件夹
    current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    save_folder = os.path.join(picture_folder, current_time)
    os.mkdir(save_folder)

    camera = cv2.VideoCapture(0)  # 读取摄像头

    frame_count = 0  # 用于给每张图片命名

    while True:
        # 读取当前帧
        ret, frame = camera.read()

        if not ret:
            print("Failed to capture frame")
            break

        # 显示摄像头画面
        # cv2.imshow("camera", frame)

        # 保存每一帧的图片
        image_path = os.path.join(save_folder, f"photo_{frame_count}.jpg")
        cv2.imwrite(image_path, frame)
        print(f"Saved {image_path}")

        frame_count += 1  # 更新帧计数器

        # 按ESC键退出
        if cv2.waitKey(5) == 27:
            break
        # 用鼠标点击窗口退出


    camera.release()
    cv2.destroyAllWindows()

detect()
