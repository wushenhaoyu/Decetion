import cv2
import os

def create_video_from_images(folder_path, output_video_path, fps=30):
    """
    从指定文件夹中的图片创建视频。
    
    参数:
    folder_path: 包含图片的文件夹路径。
    output_video_path: 输出视频的路径。
    fps: 视频帧率，默认为30。
    """
    # 构建绝对路径
    folder_path = os.path.join(os.path.dirname(__file__), folder_path)
    
    # 获取文件夹中所有图片文件
    images = [img for img in os.listdir(folder_path) if img.endswith(".png") or img.endswith(".jpg")]
    
    # 按照数字顺序排序
    images.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))
    
    # 读取第一张图片以获取尺寸信息
    if not images:
        print("No images found in the folder.")
        return
    
    image_path = os.path.join(folder_path, images[0])
    frame = cv2.imread(image_path)
    
    if frame is None:
        print(f"Failed to read the first image: {image_path}")
        return
    
    height, width, _ = frame.shape

    # 创建视频写入对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用mp4v编码器
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # 遍历所有图片并添加到视频中
    for image in images:
        image_path = os.path.join(folder_path, image)
        frame = cv2.imread(image_path)
        
        if frame is None:
            print(f"Failed to read image: {image_path}")
            continue
        
        out.write(frame)

    # 释放资源
    out.release()
    print("Video created successfully.")

if __name__ == "__main__":
    # 使用示例
    folder_path = 'MVI_40701_250_0.03'  # 替换为实际图片文件夹路径
    output_video_path = 'output_video.mp4'  # 输出视频的路径
    create_video_from_images(folder_path, output_video_path, fps=10)