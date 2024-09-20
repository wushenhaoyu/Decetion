from django.shortcuts import render
import os
from django.conf import settings
from django.conf import settings
import os
from django.conf import settings
from django.shortcuts import render
import subprocess


def index(request):
	return render(request, 'index.html')

def v_name(request, v_name):
	return render(request, 'video.html', {
		'v_name': v_name
	})





def convert_video(file_path):
    file_name = os.path.basename(file_path)
    output_path = os.path.join(settings.MEDIA_ROOT, 'converted_' + file_name)
    print(output_path)
    print(file_path)
    subprocess.run(['ffmpeg', '-i', file_path, '-c:v', 'libx264', output_path])
    return output_path



def upload_video(request):
    video_url = None  # 初始化为 None，防止首次加载时报错
    if request.method == 'POST':
        if 'video' in request.FILES:
            video = request.FILES['video']
            file_path = os.path.join(settings.MEDIA_ROOT, video.name)
            try:
                with open(file_path, 'wb') as f:
                    for chunk in video.chunks():
                        f.write(chunk)
                # 生成视频的 URL 路径，确保文件上传目录与 MEDIA_URL 配置正确
                video_url = os.path.join(settings.MEDIA_URL, video.name)
                print(f"Video uploaded to {file_path}")
            except Exception as e:
                print(f"Failed to upload video: {e}")
        else:
            print("No video file uploaded.")
    print(video_url)
    return render(request, 'upload.html', {'video_url': video_url})

from django.http import StreamingHttpResponse
import mimetypes



def video_view(request):
    # 视频文件的 URL
    video_url = '/media/show.mp4'
    return render(request, 'showvideo.html', {'video_url': video_url})