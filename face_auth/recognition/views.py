from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.parsers import MultiPartParser
from django.shortcuts import get_object_or_404
import cv2
import os
import numpy as np

from .models import Person
from .serializers import PersonSerializer
from .utils import get_face_encodings, compare_faces


class VerifyView(APIView):
    parser_classes = [MultiPartParser]

    def post(self, request):
        if 'image' not in request.FILES:
            return Response(
                {'error': 'No image provided'},
                status=status.HTTP_400_BAD_REQUEST
            )

        image = request.FILES['image']
        unknown_encodings = get_face_encodings(image)

        if not unknown_encodings:
            return Response(
                {'error': 'No faces detected'},
                status=status.HTTP_400_BAD_REQUEST
            )

        # Get the first face found
        unknown_encoding = unknown_encodings[0]

        # Compare with all known persons
        persons = Person.objects.exclude(encoding__isnull=True)
        for person in persons:
            known_encoding = np.frombuffer(person.encoding, dtype=np.float64)
            if compare_faces(known_encoding, unknown_encoding):
                return Response(
                    {'match': True, 'person': PersonSerializer(person).data},
                    status=status.HTTP_200_OK
                )

        return Response(
            {'match': False, 'message': 'No matching face found'},
            status=status.HTTP_400_BAD_REQUEST
        )


class VerifyCameraView(APIView):
    def post(self, request):
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                return Response(
                    {"error": "Camera not accessible"},
                    status=status.HTTP_400_BAD_REQUEST
                )

            # Set a timeout for camera read
            cap.set(cv2.CAP_PROP_FPS, 30)
            ret, frame = cap.read()
            cap.release()

            if not ret or frame is None:
                return Response(
                    {"error": "Failed to capture valid image frame"},
                    status=status.HTTP_400_BAD_REQUEST
                )

            # Ensure frame is 3-channel color image
            if len(frame.shape) != 3 or frame.shape[2] != 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

            # Convert to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Debug: Save frame for testing
            # cv2.imwrite('debug_frame.jpg', frame)

            unknown_encodings = get_face_encodings(rgb_frame)

            if not unknown_encodings:
                return Response(
                    {"error": "No faces detected in camera frame"},
                    status=status.HTTP_400_BAD_REQUEST
                )

            # Rest of your comparison logic...
            unknown_encoding = unknown_encodings[0]
            persons = Person.objects.exclude(encoding__isnull=True)

            for person in persons:
                known_encoding = np.frombuffer(person.encoding, dtype=np.float64)
                if compare_faces(known_encoding, unknown_encoding):
                    return Response(
                        {'match': True, 'person': PersonSerializer(person).data},
                        status=status.HTTP_200_OK
                    )

            return Response(
                {'match': False, 'message': 'No matching face found'},
                status=status.HTTP_400_BAD_REQUEST
            )

        except Exception as e:
            return Response(
                {"error": f"Camera processing failed: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class AdminUploadView(APIView):
    parser_classes = [MultiPartParser]

    def post(self, request):
        serializer = PersonSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(
                serializer.errors,
                status=status.HTTP_400_BAD_REQUEST
            )

        person = serializer.save()

        # Process the image and save encoding
        image = request.FILES['image']
        encodings = get_face_encodings(image)

        if encodings:
            # Save the first face encoding
            person.encoding = encodings[0].tobytes()
            person.save()

            return Response(
                PersonSerializer(person).data,
                status=status.HTTP_201_CREATED
            )
        else:
            person.delete()
            return Response(
                {'error': 'No faces detected in the image'},
                status=status.HTTP_400_BAD_REQUEST
            )