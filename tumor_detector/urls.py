from django.urls import path
from . import views

app_name = 'tumor_detector'

urlpatterns = [
    path('', views.home, name='home'),
    path('predict/', views.predict, name='predict'),
    path('history/', views.get_history, name='history'),
    path('reanalyze/<uuid:analysis_id>/', views.reanalyze, name='reanalyze'),
    path('report/<uuid:analysis_id>/', views.download_report, name='download_report'),
    path('segment/', views.segment, name='segment'),
    path('delete/<uuid:analysis_id>/', views.delete_analysis, name='delete_analysis'),
] 