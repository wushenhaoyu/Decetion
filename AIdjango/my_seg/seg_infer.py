import os
import sys
import cv2
from PIL import Image
import numpy as np
from my_seg.deploy.python.infer import Predictor ,parse_args,main
from my_seg.paddleseg.utils.visualize import get_pseudo_color_map
current_directory = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(current_directory)

class my_paddleseg():
    def __init__(self, args):
        self.predictor = Predictor(args)

    def run(self,img_list):
        return self.predictor.run(img_list)

class paddlesegCamera:
    def __init__(self, cfg_file=None):
        """
        初始化 my_seg.paddlesegCamera 类，加载配置文件并初始化 Predictor。
        """
        # 解析命令行参数
        self.args = parse_args()
        
        # 如果传入了自定义的配置文件路径，则使用它
        if cfg_file:
            self.args.cfg = cfg_file
        else:
            self.args.cfg = os.path.join(current_directory, 'output_inference', 'pp_liteseg_infer_model', 'deploy.yaml')
        
        # 初始化 Predictor
        self.predictor = Predictor(self.args)

    def process_frame(self, frame):
        """
        处理一帧图像并返回伪彩色输出。
        """
        output = self.predictor.run([frame])  # 输入图像列表
        pseudo_color_output = get_pseudo_color_map(output[0])  # 获取伪彩色图像
        pseudo_color_output = np.array(pseudo_color_output.convert('RGB'))  # 转换为 RGB 格式
        pseudo_color_output = cv2.cvtColor(pseudo_color_output, cv2.COLOR_RGB2BGR)  # 转换为 BGR 格式
        return pseudo_color_output

    def run(self):
        """
        启动摄像头，实时显示处理后的图像。
        """
        # 打开摄像头
        camera = cv2.VideoCapture(0)
        
        while True:
            # 读取一帧图像
            _, frame = camera.read()
            
            if frame is None:
                break

            # 处理当前帧
            processed_frame = self.process_frame(frame)

            # 显示处理后的图像
            cv2.imshow("Segmentation Result", processed_frame)

            # 按 'q' 键退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # 释放摄像头资源并关闭所有 OpenCV 窗口
        camera.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    args = parse_args()
    args.cfg = os.path.join(current_directory,'output_inference','pp_liteseg_infer_model','deploy.yaml')
    predictor = Predictor(args)
    camera = cv2.VideoCapture(0)
    while True:
        _, frame = camera.read()
        output = predictor.run([frame])#里面放图片的列表
        pseudo_color_output = get_pseudo_color_map(output[0])
        #pseudo_color_output.save("output.png")
        pseudo_color_output = np.array(pseudo_color_output.convert('RGB'))
            # 将 RGB 图像转换为 BGR 图像
        pseudo_color_output = cv2.cvtColor(pseudo_color_output, cv2.COLOR_RGB2BGR)
            
            # 显示图像
        cv2.imshow("res", pseudo_color_output)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()