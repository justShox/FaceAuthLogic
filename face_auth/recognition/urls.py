from django.urls import path
from .views import VerifyCameraView, VerifyImageView

urlpatterns = [
    path('verify_camera/', VerifyCameraView.as_view(), name='verify_camera'),
    path('verify/', VerifyImageView.as_view(), name='verify'),
]