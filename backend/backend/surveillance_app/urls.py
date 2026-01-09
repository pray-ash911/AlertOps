from django.urls import path
from . import views
# Assuming get_latest_status is defined in views.py, we can typically rely on the 'from . import views'
# But since you explicitly imported it, we'll keep it for clarity.
from .views import get_latest_status

urlpatterns = [

    # Existing: GET endpoint for real-time video streaming
    path('video_feed/', views.video_feed_view, name='video_feed'),

    # CORRECTED PATH: Status API Endpoint now includes /api/
    path('api/latest_status/', get_latest_status, name='latest_status'),

    # Home route for basic check
    path('', views.home, name='home'),

    # CORRECTED PATH: Endpoint for fetching logged events now includes /api/
    # This matches the Streamlit request: http://127.0.0.1:8000/api/logs/
    path('api/logs/', views.event_logs_view, name='event_logs'),

    # Analytics endpoint for dashboard data
    path('api/analytics/', views.analytics_view, name='analytics'),

    # Authentication API endpoints
    path('api/register/', views.register_view, name='register'),
    path('api/login/', views.login_view, name='login'),
    path('api/logout/', views.logout_view, name='logout'),
    path('api/current-user/', views.current_user_view, name='current_user'),

    # Frontend pages
    path('login/', views.login_page, name='login_page'),
    path('register/', views.register_page, name='register_page'),
]
