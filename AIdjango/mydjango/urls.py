from django.contrib import admin
from django.urls import path
from user.views import opencam, closecam,initialize
from video.views import index, v_name,upload_video,video_view
from user.gen_display import video,initialize
from django.conf.urls import include
from django.conf import settings
from django.conf.urls.static import static
urlpatterns = [
    path('admin/', admin.site.urls),
    path('opencam/', opencam),
    path('closecam/', closecam),
    path("",initialize),
    path('video/', index),  # 匹配 /video/
    path('video/<str:v_name>/', v_name, name='v_name'),  # 匹配 /video/<v_name>/,打开摄像头之后进入http://127.0.0.1:8000/video/wms/就可以看到了
    path('upload/', upload_video, name='upload_video'),
    # video/是播放视频的页面
    path('showvideo/', video_view),
    path('livedisplay', video),
    path("",initialize)
]


urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
