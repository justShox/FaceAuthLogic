from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.parsers import MultiPartParser
import cv2
import numpy as np
from .utils import FaceRecognizer

face_recognizer = FaceRecognizer()


class VerifyCameraView(APIView):
    def post(self, request):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return Response(
                {"error": "Camera not accessible"},
                status=status.HTTP_400_BAD_REQUEST
            )

        ret, frame = cap.read()
        cap.release()

        if not ret:
            return Response(
                {"error": "Failed to capture image"},
                status=status.HTTP_400_BAD_REQUEST
            )

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        result = face_recognizer.recognize_faces(gray)

        if "error" in result:
            return Response(result, status=status.HTTP_400_BAD_REQUEST)
        return Response(result)


class VerifyImageView(APIView):
    parser_classes = [MultiPartParser]

    def post(self, request):
        if 'image' not in request.FILES:
            return Response(
                {"error": "No image provided. Use multipart/form-data with 'image' field"},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            image = request.FILES['image']
            if not image.name.lower().endswith(('.jpg', '.jpeg', '.png')):
                return Response(
                    {"error": "Only JPG/PNG images supported"},
                    status=status.HTTP_400_BAD_REQUEST
                )

            result = face_recognizer.recognize_faces(image)
            return Response(result)

        except Exception as e:
            return Response(
                {"error": f"Processing failed: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )