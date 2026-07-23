from django.urls import path
from . import views

urlpatterns = [
    path('jobs/<uuid:pk>/analyze/', views.AnalyzeView.as_view()),
    path('jobs/<uuid:pk>/report/', views.ReportDownloadView.as_view()),
]
