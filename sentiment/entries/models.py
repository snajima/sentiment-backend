from django.contrib.auth.models import User
from django.db import models


class Entry(models.Model):
    # many to one relationship with user
    user = models.ForeignKey(User, on_delete=models.SET_NULL, null=True)
    entry_description = models.TextField(default=None)
    emotion = models.CharField(max_length=20, default=None)