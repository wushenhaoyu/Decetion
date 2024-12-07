import os
import sys
import cv2
import numpy as np
from PIL import Image
from deploy12.python.infer import Predictor, parse_args
from paddleseg12.utils.visualize import get_pseudo_color_map

# 确保当前路径被添加到 sys.path
current_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_directory)

class PaddleSegCamera:
    def __init__(self, cfg_file=None):
        """
        初始化 PaddleSegCamera 类，加载配置文件并初始化 Predictor。
        """
        # 解析命令行参数
        self.args = parse_args()
        
        # 如果传入了自定义的配置文件路径，则使用它
        if cfg_file:
            self.args.cfg = cfg_file
        else:
            self.args.cfg = os.path.join(current_directory, 'output_inference12', 'pp_liteseg_infer_model', 'deploy.yaml')
        
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

# 作为模块运行时
if __name__ == '__main__':
    # 创建 PaddleSegCamera 实例并运行
    paddle_seg_camera = PaddleSegCamera()
    paddle_seg_camera.run()
