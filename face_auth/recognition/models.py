from django.db import models
import os

def person_image_path(instance, filename):
    return f'persons/{instance.name}/{filename}'

class Person(models.Model):
    name = models.CharField(max_length=100)
    image = models.ImageField(upload_to=person_image_path)
    encoding = models.BinaryField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name