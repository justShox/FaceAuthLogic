from django.urls import path
from .views import VerifyView, VerifyCameraView, AdminUploadView

urlpatterns = [
    path('verify/', VerifyView.as_view(), name='verify'),
    path('verify_camera/', VerifyCameraView.as_view(), name='verify_camera'),
    path('admin/upload/', AdminUploadView.as_view(), name='admin_upload'),
]