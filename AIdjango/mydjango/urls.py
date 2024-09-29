from django.contrib import admin
from django.urls import path
from user.gen_display import video,upload_video,upload_photo,ConfirmParams, close_camera, open_camera,getAllRecordFile,get_progress
from user.gen_display import initialize,getAllPhotoFile,getAllVideoFile,getAllCam, Camchoice,video_record_on,video_record_off,stream_record_download
from user.gen_display import stream_video_download,stream_photo_download,stream_video,stream_photo
from django.conf import settings
from django.urls import path, re_path
from django.conf import settings
from django.views.static import serve
from django.conf.urls.static import static
urlpatterns = [
    path('admin/', admin.site.urls),
    path("",initialize) ,        #初始化深度学习模型
    path("ConfirmParams",ConfirmParams),
    path("getAllCam",getAllCam),#得到所有的摄像头设备
    path("Camchoice",Camchoice),#给出使用的摄像头
    path('opencam', open_camera),     #打开摄像头
    path('closecam',  close_camera),     #关闭摄像头
    path('livedisplay', video),     #实时演示功能
    path('video_record_on', video_record_on),     #开启录制
    path('video_record_off',  video_record_off),    #关闭录制
    path("getAllRecordFile",getAllRecordFile),#得到所有的录制文件
    path('uploadVideo', upload_video, name='upload_video'),#上传视频
    path("get_progress",get_progress),#得到视频处理的进度条
    path("stream_record_download",stream_record_download),#下载录制的视频。
    path("stream_photo_download",stream_photo_download),#下载录制的视频。
    path("stream_video_download",stream_video_download),#下载录制的视频。
    path('upload_photo', upload_photo, name='upload_photo'),#上传照片
    path("stream_video",stream_video),
    path("stream_photo",stream_photo),



    path("getAllPhotoFile",getAllPhotoFile),#得到所有的照片文件
    path("getAllVideoFile",getAllVideoFile),#得到所有的视频文件


]


urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
