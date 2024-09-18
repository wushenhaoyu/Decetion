from django.contrib import admin
from django.urls import path
from user.views import opencam, closecam
from video.views import index, v_name
from django.conf.urls import include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('opencam/', opencam),
    path('closecam/', closecam),
    path('video/', index),  # 匹配 /video/
    path('video/<str:v_name>/', v_name, name='v_name'),  # 匹配 /video/<v_name>/
    path('video/<str:v_name>/', v_name, name='v_name'),  # 匹配 /video/<v_name>/
]
