from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    
    # path('filter_poses/', views.filter_poses, name='filter_poses'),  
    path('', views.home, name='home'), 
    path('pose/', views.yoga_poses, name='yoga_poses'),
    # path('analyze/', views.analyze_pose_view, name='analyze_pose_view'),
    path('upload/', views.upload_image_for_pose, name='upload_image_for_pose'),
    # path('realtime/', views.realtime_pose_base, name='realtime_pose_base'),
    path('yoga_details/<str:pose_name>/', views.yoga_details, name='yoga_details'),
    # path('pose/<str:pose_name>/', views.yoga_options, name='yoga_options'), 
    # path('pose/<str:pose_name>/upload/', views.show_views, name='show_views'), 
    path('pose/<str:pose_name>/realtime/', views.live_correction, name='live_correction'),
    path('pose/<str:pose_name>/yoga_views/', views.yoga_views, name='yoga_views'),  
    path('pose/<str:pose_name>/upload/', views.upload_image, name='upload_image'),  
    path('pose/<str:pose_name>/analyze/', views.analyze_pose, name='analyze_pose'),  
    path('live/', views.live_stream, name='live_stream'),
    path('stop_stream/', views.stop_stream, name='stop_stream'),

]
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    