from django.contrib import admin
from django.urls import path
from user.gen_display import video,upload_video,video_view,upload_photo,ConfirmParams,photo_view, close_camera
from user.gen_display import initialize
from django.conf import settings
from django.conf.urls.static import static
urlpatterns = [
    path('admin/', admin.site.urls),
    # path('opencam/', opencam),
    # path('closecam/', closecam),
    # path("",initialize),
    # path('video/', index),  # 匹配 /video/
    # path('video/<str:v_name>/', v_name, name='v_name'),  # 匹配 /video/<v_name>/,打开摄像头之后进入http://127.0.0.1:8000/video/wms/就可以看到了
    path('uploadVideo', upload_video, name='upload_video'),#上传视频
    path('uploadPhoto', upload_photo, name='upload_photo'),#上传照片
    # video/是播放视频的页面
    path('showvideo', video_view),      #展示视频
    path('showphoto', photo_view),      #展示视频
    path('opencam', video),     #实时演示功能
    path('closecam',  close_camera),     #实时演示功能
    path("",initialize) ,        #初始化深度学习模型
    path("ConfirmParams",ConfirmParams)
]


urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
