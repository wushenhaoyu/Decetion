import cv2
import numpy as np
from haze.test_real import HazeRemover
from PIL import Image
if __name__ == '__main__':


    haze_remover = HazeRemover()
    
    # 示例: 处理单帧图像
    frame_path = r"C:\Users\cat\Desktop\haze_source\input\HF_Google_283.png"
    frame = Image.open(frame_path)
    
    processed_frame = haze_remover.haze_frame(frame)

    if processed_frame.ndim == 3 and processed_frame.shape[2] == 3:
        processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR)
    cv2.imshow('Processed Frame', processed_frame)
    cv2.waitKey(0)  # 等待按键事件
    cv2.destroyAllWindows()
    

