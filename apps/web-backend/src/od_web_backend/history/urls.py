from django.urls import path
from . import views

urlpatterns = [
    path('history/', views.HistoryListView.as_view()),
    path('history/<uuid:pk>/', views.JobDetailView.as_view()),
    path('dashboard/stats/', views.DashboardStatsView.as_view()),
]
