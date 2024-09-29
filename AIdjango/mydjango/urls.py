from django.contrib import admin
from django.urls import path
from user.gen_display import video,upload_video,video_view,upload_photo,ConfirmParams,photo_view, close_camera, open_camera,video_record,close_camera_record,getAllRecordFile,get_progress
from user.gen_display import initialize,getAllPhotoFile,getAllVideoFile,getAllCam, Camchoice
from django.conf import settings
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
    path('livedisplayRecord', video_record),     #录制演示功能外加保存
    path('closecamRecord',  close_camera_record),    #关闭录制
    path("getAllRecordFile",getAllRecordFile),#得到所有的录制文件

    path('uploadVideo', upload_video, name='upload_video'),#上传视频
    path("get_progress",get_progress),#得到视频处理的进度条

    path('uploadPhoto', upload_photo, name='upload_photo'),#上传照片
    
    path("getAllPhotoFile",getAllPhotoFile),#得到所有的照片文件
    path("getAllVideoFile",getAllVideoFile)#得到所有的视频文件

]


urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
