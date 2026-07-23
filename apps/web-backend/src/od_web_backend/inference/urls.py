from django.urls import path
from . import views

urlpatterns = [
    path('models/', views.ModelListView.as_view()),
    path('models/sync/', views.ModelSyncView.as_view()),
    path('detect/single/', views.DetectSingleView.as_view()),
    path('detect/batch/', views.DetectBatchView.as_view()),
]
