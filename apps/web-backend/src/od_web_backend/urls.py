from django.conf import settings
from django.urls import path, include
from django.views.static import serve as static_serve


urlpatterns = [
    path('api/', include('od_web_backend.core.urls')),
    path('api/', include('od_web_backend.inference.urls')),
    path('api/', include('od_web_backend.history.urls')),
    path('api/', include('od_web_backend.analysis.urls')),
]

# 开发环境由 Django 直接服务 media(生产交给 Nginx)
if settings.DEBUG:
    urlpatterns += [
        path('media/<path:path>', static_serve, {'document_root': settings.MEDIA_ROOT}),
    ]
