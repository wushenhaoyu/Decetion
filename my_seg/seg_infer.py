import os
import sys
import cv2
from PIL import Image
import numpy as np
from deploy.python.infer import Predictor ,parse_args,main
from my_seg.paddleseg.utils.visualize import get_pseudo_color_map
current_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_directory)

class my_paddleseg():
    def __init__(self, args):
        self.predictor = Predictor(args)

    def run(self,img_list):
        return self.predictor.run(img_list)


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