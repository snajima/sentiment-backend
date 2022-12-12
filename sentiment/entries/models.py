from django.contrib.auth.models import User
from django.db import models
from persons.models import Person

class Entry(models.Model):
    # many to one relationship with user
    poster = models.ForeignKey(
        Person, on_delete=models.CASCADE, default=None, related_name="poster"
    )
    entry_description = models.TextField(default=None)
    emotion = models.CharField(max_length=20, default=None)
    date = models.DateField(null=False)