# video/views.py
from django.shortcuts import render

# Create your views here.
def index(request):
	return render(request, 'index.html')

def v_name(request, v_name):
	return render(request, 'video.html', {
		'v_name': v_name
	})
